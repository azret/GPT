using System;
using System.IO;
using System.Text;

using static std;

unsafe class GPT {
    // ----------------------------------------------------------------------------
    // all the individual layers' forward and backward passes
    // B = batch_size, T = sequence_length, C = channels, V = vocab_size
    static unsafe void encoder_forward(float* outp,
                       int* inp, float* wte, float* wpe,
                       int B, int T, int C) {
        // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // seek to the output position in out[b,t,:]
                float* out_bt = outp + b * T * C + t * C;
                // get the index of the token at inp[b, t]
                int ix = inp[b * T + t];
                // seek to the position in wte corresponding to the token
                float* wte_ix = wte + ix * C;
                // seek to the position in wpe corresponding to the position
                float* wpe_t = wpe + t * C;
                // add the two vectors and store the result in out[b,t,:]
                for (int i = 0; i < C; i++) {
                    out_bt[i] = wte_ix[i] + wpe_t[i];
                }
            }
        }
    }

    static unsafe void layernorm_forward(float* outp, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C) {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both inp and out are (B,T,C) of the activations
        // mean and rstd are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // seek to the input position inp[b,t,:]
                float* x = inp + b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < C; i++) {
                    m += x[i];
                }
                m = m / C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < C; i++) {
                    float xshift = x[i] - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // calculate the rstd (reciprocal standard deviation)
                float s = 1.0f / sqrtf(v + eps);
                // seek to the output position in out[b,t,:]
                float* out_bt = outp +b * T * C + t * C;
                for (int i = 0; i < C; i++) {
                    float n = (s * (x[i] - m)); // normalize
                    float o = n * weight[i] + bias[i]; // scale and shift
                    out_bt[i] = o; // write
                }
                // cache the mean and rstd for the backward pass later
                mean[b * T + t] = m;
                rstd[b * T + t] = s;
            }
        }
    }

    static unsafe void matmul_forward(float* outp,
                         float* inp, float* weight, float* bias,
                         int B, int T, int C, int OC) {
        // the most naive implementation of matrix multiplication
        // this serves as an algorithmic reference, and as a fallback for
        // unfriendly input shapes inside matmul_forward(), below.
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int bt = b * T + t;
                for (int o = 0; o < OC; o++) {
                    float val = (bias != null) ? bias[o] : 0.0f;
                    for (int i = 0; i < C; i++) {
                        val += inp[bt * C + i] * weight[o * C + i];
                    }
                    outp[bt * OC + o] = val;
                }
            }
        }
    }

    static unsafe void attention_forward(float* outp, float* preatt, float* att,
                           float* inp,
                           int B, int T, int C, int NH) {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / sqrtf(hs);

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < NH; h++) {
                    float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                    float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                    float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++) {
                        float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++) {
                            val += query_t[i] * key_t2[i];
                        }
                        val *= scale;
                        if (val > maxval) {
                            maxval = val;
                        }

                        preatt_bth[t2] = val;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float expv = expf(preatt_bth[t2] - maxval);
                        expsum += expv;
                        att_bth[t2] = expv;
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++) {
                        if (t2 <= t) {
                            att_bth[t2] *= expsum_inv;
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att_bth[t2] = 0.0f;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    float* out_bth = outp + b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                    for (int t2 = 0; t2 <= t; t2++) {
                        float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float att_btht2 = att_bth[t2];
                        for (int i = 0; i < hs; i++) {
                            out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
            }
        }
    }

    static unsafe void residual_forward(float* outp, float* inp1, float* inp2, int N) {
        for (int i = 0; i < N; i++) {
            outp[i] = inp1[i] + inp2[i];
        }
    }

    static readonly float GELU_SCALING_FACTOR = sqrtf(2.0f / MathF.PI);

    static unsafe void gelu_forward(float* outp, float* inp, int N) {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for (int i = 0; i < N; i++) {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            outp[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
        }
    }

    static unsafe void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
        // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,Vp) of the unnormalized log probabilities
        // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
        // example: Vp is 50304 and V is 50257
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // probs <- softmax(logits)
                float* logits_bt = logits + b * T * Vp + t * Vp;
                float* probs_bt = probs + b * T * Vp + t * Vp;

                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better
                for (int i = 0; i < V; i++) {
                    if (logits_bt[i] > maxval) {
                        maxval = logits_bt[i];
                    }
                }
                float sum = 0.0f;
                for (int i = 0; i < V; i++) {
                    probs_bt[i] = expf(logits_bt[i] - maxval);
                    sum += probs_bt[i];
                }
                // note we only loop to V, leaving the padded dimensions
                for (int i = 0; i < V; i++) {
                    probs_bt[i] /= sum;
                }
                // for extra super safety we may wish to include this too,
                // forcing the probabilities here to be zero, but it shouldn't matter
                for (int i = V; i < Vp; i++) {
                    probs_bt[i] = 0.0f;
                }
            }
        }
    }


    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack = 1)]
    public struct GPT2Config {
        public int max_seq_len; // max sequence length, e.g. 1024
        public int vocab_size; // vocab size, e.g. 50257
        public int padded_vocab_size; // padded to e.g. %128==0, 50304
        public int num_layers; // number of layers, e.g. 12
        public int num_heads; // number of heads in attention, e.g. 12
        public int channels; // number of channels, e.g. 768
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack = 1)]
    public unsafe struct ParameterTensors {
        public const int NUM_PARAMETER_TENSORS = 16;
        public float* wte; // (V, C)
        public float* wpe; // (maxT, C)
        public float* ln1w; // (L, C)
        public float* ln1b; // (L, C)
        public float* qkvw; // (L, 3*C, C)
        public float* qkvb; // (L, 3*C)
        public float* attprojw; // (L, C, C)
        public float* attprojb; // (L, C)
        public float* ln2w; // (L, C)
        public float* ln2b; // (L, C)
        public float* fcw; // (L, 4*C, C)
        public float* fcb; // (L, 4*C)
        public float* fcprojw; // (L, C, 4*C)
        public float* fcprojb; // (L, C)
        public float* lnfw; // (C)
        public float* lnfb; // (C)
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack = 1)]
    public unsafe struct ActivationTensors {
        public const int NUM_ACTIVATION_TENSORS = 23;
        public float* encoded; // (B, T, C)
        public float* ln1; // (L, B, T, C)
        public float* ln1_mean; // (L, B, T)
        public float* ln1_rstd; // (L, B, T)
        public float* qkv; // (L, B, T, 3*C)
        public float* atty; // (L, B, T, C)
        public float* preatt; // (L, B, NH, T, T)
        public float* att; // (L, B, NH, T, T)
        public float* attproj; // (L, B, T, C)
        public float* residual2; // (L, B, T, C)
        public float* ln2; // (L, B, T, C)
        public float* ln2_mean; // (L, B, T)
        public float* ln2_rstd; // (L, B, T)
        public float* fch; // (L, B, T, 4*C)
        public float* fch_gelu; // (L, B, T, 4*C)
        public float* fcproj; // (L, B, T, C)
        public float* residual3; // (L, B, T, C)
        public float* lnf; // (B, T, C)
        public float* lnf_mean; // (B, T)
        public float* lnf_rstd; // (B, T)
        public float* logits; // (B, T, V)
        public float* probs; // (B, T, V)
        public float* losses; // (B, T)
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack = 1)]
    public unsafe struct GPT2 {
        public GPT2Config config;
        // the weights (parameters) of the model, and their sizes
        public ParameterTensors _params;
        public fixed int param_sizes[ParameterTensors.NUM_PARAMETER_TENSORS];
        public float* params_memory;
        public int num_parameters;
        // gradients of the weights
        public ParameterTensors grads;
        public float* grads_memory;
        // buffers for the AdamW optimizer
        public float* m_memory;
        public float* v_memory;
        // the activations of the model, and their sizes
        public ActivationTensors acts;
        public fixed int act_sizes[ActivationTensors.NUM_ACTIVATION_TENSORS];
        public float* acts_memory;
        public int num_activations;
        // gradients of the activations
        public ActivationTensors grads_acts;
        public float* grads_acts_memory;
        // other run state configuration
        public int batch_size; // the batch size (B) of current forward pass
        public int seq_len; // the sequence length (T) of current forward pass
        public int* inputs; // the input tokens for the current forward pass
        public int* targets; // the target tokens for the current forward pass
        public float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    }

    unsafe static void fill_in_parameter_sizes(int* param_sizes, GPT2Config config) {
        int Vp = config.padded_vocab_size;
        int C = config.channels;
        int maxT = config.max_seq_len;
        int L = config.num_layers;
        param_sizes[0] = Vp * C; // wte
        param_sizes[1] = maxT * C; // wpe
        param_sizes[2] = L * C; // ln1w
        param_sizes[3] = L * C; // ln1b
        param_sizes[4] = L * (3 * C) * C; // qkvw
        param_sizes[5] = L * (3 * C); // qkvb
        param_sizes[6] = L * C * C; // attprojw
        param_sizes[7] = L * C; // attprojb
        param_sizes[8] = L * C; // ln2w
        param_sizes[9] = L * C; // ln2b
        param_sizes[10] = L * (4 * C) * C; // fcw
        param_sizes[11] = L * (4 * C); // fcb
        param_sizes[12] = L * C * (4 * C); // fcprojw
        param_sizes[13] = L * C; // fcprojb
        param_sizes[14] = C; // lnfw
        param_sizes[15] = C; // lnfb
    }

    // allocate memory for the parameters and point the individual tensors to the right places
    unsafe static float* malloc_and_point_parameters(ParameterTensors* _params, int* param_sizes) {
        ulong num_parameters = 0;
        for (int i = 0; i < ParameterTensors.NUM_PARAMETER_TENSORS; i++) {
            num_parameters += (ulong)param_sizes[i];
        }
        // malloc all parameters all at once
        float* params_memory = (float*)malloc(num_parameters * sizeof(float));
        // assign all the tensors
        float**[] ptrs = {
            &_params->wte, &_params->wpe, &_params->ln1w, &_params->ln1b, &_params->qkvw, &_params->qkvb,
            &_params->attprojw, &_params->attprojb, &_params->ln2w, &_params->ln2b, &_params->fcw, &_params->fcb,
            &_params->fcprojw, &_params->fcprojb, &_params->lnfw, &_params->lnfb
        };
        float* params_memory_iterator = params_memory;
        for (int i = 0; i < ParameterTensors.NUM_PARAMETER_TENSORS; i++) {
            *(ptrs[i]) = params_memory_iterator;
            params_memory_iterator += param_sizes[i];
        }
        return params_memory;
    }

    unsafe static void gpt2_build_from_checkpoint(GPT2* model, string checkpoint_path) {

        // read in model from a checkpoint file
        var model_file = fopen(checkpoint_path, "rb");
        int* model_header = stackalloc int[256];
        fread(model_header, sizeof(int), 256, model_file);
        if (model_header[0] != 20240326) {
            printf("Bad magic model file\n"); exit(1);
        }
        if (model_header[1] != 3) {
            printf("Bad version in model file\n");
            printf("---> HINT: try to re-run `python train_gpt2.py`\n");
            exit(1);
        }

        // read in hyperparameters
        int maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
        model->config.max_seq_len = maxT = model_header[2];
        model->config.vocab_size = V = model_header[3];
        model->config.num_layers = L = model_header[4];
        model->config.num_heads = NH = model_header[5];
        model->config.channels = C = model_header[6];
        model->config.padded_vocab_size = Vp = model_header[7];
        printf("[GPT-2]\n");
        printf("max_seq_len: %zu\n", maxT);
        printf("vocab_size: %zu\n", V);
        printf("padded_vocab_size: %zu\n", Vp);
        printf("num_layers: %zu\n", L);
        printf("num_heads: %zu\n", NH);
        printf("channels: %zu\n", C);

        // allocate space for all the parameters and read them in
        fill_in_parameter_sizes(model->param_sizes, model->config);

        // count the number of parameters
        int num_parameters = 0;
        for (int i = 0; i < ParameterTensors.NUM_PARAMETER_TENSORS; i++) {
            num_parameters += model->param_sizes[i];
        }
        printf("num_parameters: %zu\n", num_parameters);
        model->num_parameters = num_parameters;

        // read in all the parameters from file
        model->params_memory = malloc_and_point_parameters(&model->_params, model->param_sizes);
        fread(model->params_memory, sizeof(float), num_parameters, model_file);

        fclose(model_file);

        // other inits
        model->acts_memory = null;
        model->grads_memory = null;
        model->m_memory = null;
        model->v_memory = null;
        model->grads_acts_memory = null;
        model->inputs = null;
        model->targets = null;
        model->batch_size = 0;
        model->seq_len = 0;
        model->mean_loss = -1.0f; // -1.0f will designate no loss
    }

    unsafe static void fill_in_activation_sizes(int* act_sizes, GPT2Config config, int B, int T) {
        int C = config.channels;
        int NH = config.num_heads;
        int L = config.num_layers;
        int Vp = config.padded_vocab_size;
        act_sizes[0] = B * T * C; // encoded
        act_sizes[1] = L * B * T * C; // ln1
        act_sizes[2] = L * B * T; // ln1_mean
        act_sizes[3] = L * B * T; // ln1_rstd
        act_sizes[4] = L * B * T * 3 * C; // qkv
        act_sizes[5] = L * B * T * C; // atty
        act_sizes[6] = L * B * NH * T * T; // preatt
        act_sizes[7] = L * B * NH * T * T; // att
        act_sizes[8] = L * B * T * C; // attproj
        act_sizes[9] = L * B * T * C; // residual2
        act_sizes[10] = L * B * T * C; // ln2
        act_sizes[11] = L * B * T; // ln2_mean
        act_sizes[12] = L * B * T; // ln2_rstd
        act_sizes[13] = L * B * T * 4 * C; // fch
        act_sizes[14] = L * B * T * 4 * C; // fch_gelu
        act_sizes[15] = L * B * T * C; // fcproj
        act_sizes[16] = L * B * T * C; // residual3
        act_sizes[17] = B * T * C; // lnf
        act_sizes[18] = B * T; // lnf_mean
        act_sizes[19] = B * T; // lnf_rstd
        act_sizes[20] = B * T * Vp; // logits
        act_sizes[21] = B * T * Vp; // probs
        act_sizes[22] = B * T; // losses
    }

    unsafe static float* malloc_and_point_activations(ActivationTensors* acts, int* act_sizes) {
        int num_activations = 0;
        for (int i = 0; i < ActivationTensors.NUM_ACTIVATION_TENSORS; i++) {
            num_activations += act_sizes[i];
        }
        float* acts_memory = (float*)malloc(num_activations, sizeof(float));
        float**[] ptrs = {
            &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
            &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
            &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
            &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
        };
        float* acts_memory_iterator = acts_memory;
        for (int i = 0; i < ActivationTensors.NUM_ACTIVATION_TENSORS; i++) {
            *(ptrs[i]) = acts_memory_iterator;
            acts_memory_iterator += act_sizes[i];
        }
        return acts_memory;
    }

    private static void exit(int v) {
        throw new NotImplementedException();
    }

    unsafe static void gpt2_forward(GPT2* model, int* inputs, int* targets, int B, int T) {
        // targets are optional and could be NULL

        // ensure the model was initialized or error out
        if (model->params_memory == null) {
            printf("Error: model was not initialized properly.\n");
            exit(1);
        }

        // convenience parameters (size_t to help prevent int overflow)
        int V = model->config.vocab_size;
        int Vp = model->config.padded_vocab_size;
        int L = model->config.num_layers;
        int NH = model->config.num_heads;
        int C = model->config.channels;

        // validate inputs, all indices must be in the range [0, V)
        for (int i = 0; i < B * T; i++) {
            Assert<InvalidOperationException>.That(0 <= inputs[i] && inputs[i] < V);
            if (targets != null) {
                Assert<InvalidOperationException>.That(0 <= targets[i] && targets[i] < V);
            }
        }

        // allocate space for all the activations if needed (done here, lazily)
        if (model->acts_memory == null) {
            // record the current B,T as well
            model->batch_size = B;
            model->seq_len = T;
            // and now allocate the space
            fill_in_activation_sizes(model->act_sizes, model->config, B, T);
            int num_activations = 0;
            for (int i = 0; i < ActivationTensors.NUM_ACTIVATION_TENSORS; i++) {
                num_activations += model->act_sizes[i];
            }
            printf("num_activations: %zu\n", num_activations);
            model->num_activations = num_activations;
            model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
            // also create memory for caching inputs and targets
            model->inputs = (int*)malloc(B * T, sizeof(int));
            model->targets = (int*)malloc(B * T, sizeof(int)); // might be unused if we never have targets but it's small
        } else {
            // validate B,T is consistent with how we've allocated the memory before
            // in principle we could get more clever here in the future, for now this is safest
            if (B != model->batch_size || T != model->seq_len) {
                printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
                exit(1);
            }
        }

        // cache the inputs/targets
        kernel32.CopyMemory(model->inputs, inputs, (ulong)B * (ulong)T * (ulong)sizeof(int));
        
        if (targets != null) {
            kernel32.CopyMemory(model->targets, targets, (ulong)B * (ulong)T * (ulong)sizeof(int));
        }

        // forward pass
        ParameterTensors _params = model->_params; // for brevity
        ActivationTensors acts = model->acts;
        float* residual;
        encoder_forward(acts.encoded, inputs, _params.wte, _params.wpe, B, T, C); // encoding goes into residual[0]
        for (int l = 0; l < L; l++) {

            residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            float* l_ln1w = _params.ln1w + l * C;
            float* l_ln1b = _params.ln1b + l * C;
            float* l_qkvw = _params.qkvw + l * 3 * C * C;
            float* l_qkvb = _params.qkvb + l * 3 * C;
            float* l_attprojw = _params.attprojw + l * C * C;
            float* l_attprojb = _params.attprojb + l * C;
            float* l_ln2w = _params.ln2w + l * C;
            float* l_ln2b = _params.ln2b + l * C;
            float* l_fcw = _params.fcw + l * 4 * C * C;
            float* l_fcb = _params.fcb + l * 4 * C;
            float* l_fcprojw = _params.fcprojw + l * C * 4 * C;
            float* l_fcprojb = _params.fcprojb + l * C;

            // get the pointers of the activations for this layer
            float* l_ln1 = acts.ln1 + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            float* l_qkv = acts.qkv + l * B * T * 3 * C;
            float* l_atty = acts.atty + l * B * T * C;
            float* l_preatt = acts.preatt + l * B * NH * T * T;
            float* l_att = acts.att + l * B * NH * T * T;
            float* l_attproj = acts.attproj + l * B * T * C;
            float* l_residual2 = acts.residual2 + l * B * T * C;
            float* l_ln2 = acts.ln2 + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch = acts.fch + l * B * T * 4 * C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
            float* l_fcproj = acts.fcproj + l * B * T * C;
            float* l_residual3 = acts.residual3 + l * B * T * C;

            // now do the forward pass
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, _params.lnfw, _params.lnfb, B, T, C);
        matmul_forward(acts.logits, acts.lnf, _params.wte, null, B, T, C, Vp);
        softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

        // also forward the cross-entropy loss function if we have the targets
        if (targets != null) {
            // crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
            // for convenience also evaluate the mean loss
            float mean_loss = 0.0f;
            for (int i = 0; i < B * T; i++) { mean_loss += model->acts.losses[i]; }
            mean_loss /= B * T;
            model->mean_loss = mean_loss;
        } else {
            // if we don't have targets, we don't have a loss
            model->mean_loss = -1.0f;
        }
    }

    public class Tokenizer {
        public uint vocab_size;
        public byte[][] token_table;
        public bool init_ok;
        public int eot_token; // <|endoftext|> token id
    }

    unsafe static void tokenizer_init(Tokenizer tokenizer, string filename) {
        var file = fopen(filename, "rb");
        if (file == IntPtr.Zero) {
            // try to be more helpful as we just added this feature, erase later
            printf("---\n");
            printf("WARNING: Failed to open the tokenizer file %s\n", filename);
            printf("The Tokenizer is a new feature added April 14 2024.\n");
            printf("Re-run `python train_gpt2.py` to write it\n");
            printf("---\n");
            tokenizer.init_ok = false;
            return;
        }
        // read in the header
        uint* header = stackalloc uint[256];
        fread(header, sizeof(uint), 256, file);
        Assert<InvalidDataException>.That(header[0] == 20240328);
        uint version = header[1];
        tokenizer.vocab_size = header[2];
        if (version == 1) {
            // version 1 didn't include the EOT token id
            // so we assume it is 50256, the EOT in GPT-2
            Assert<InvalidDataException>.That(tokenizer.vocab_size == 50257); // let's be defensive here
            tokenizer.eot_token = 50256;
        } else if (version == 2) {
            tokenizer.eot_token = (int)header[3];
        } else {
            printf("Tokenizer model file %s has bad version: %d\n", filename, version);
            exit(1);
        }
        // read in all the tokens
        byte length;
        tokenizer.token_table = new byte[tokenizer.vocab_size][]; // (byte**)malloc((ulong)tokenizer->vocab_size * (ulong)sizeof(byte));
        for (int i = 0; i < tokenizer.vocab_size; i++) {
            fread(&length, sizeof(byte), 1, file);
            Assert<InvalidDataException>.That(length > 0); // every token should be at least one character
            byte[] token_bytes = new byte[length + 1];
            fread(token_bytes, length, file);
            token_bytes[length] = (byte)'\0';  // Add null terminator for printing
            tokenizer.token_table[i] = token_bytes;
        }
        // cleanups
        fclose(file);
        tokenizer.init_ok = true;
    }

    public static byte[] tokenizer_decode(Tokenizer tokenizer, int token_id) {
        if (!tokenizer.init_ok) {
            return null;
        }
        if (token_id < tokenizer.vocab_size) {
            return tokenizer.token_table[token_id];
        } else {
            Console.Write("invalid token id %d!\n", token_id);
            return null;
        }
    }

    static void safe_printf(byte[] piece) {
        // the tokens are raw bytes, and we we only want to print the printable ones
        // many bytes can be various control codes, backspace, etc.
        if (piece == null) { return; }
        if (piece[0] == '\0') { return; }
        // handle individual byte tokens
        // every token is asserted to be at least one byte so doing piece[1] is ok
        if (piece[1] == '\0') {
            byte byte_val = (byte)piece[0];
            if (!(isprint(byte_val) || isspace(byte_val))) {
                return; // weird byte, don't print it
            }
        }
        Console.Write("{0}", Encoding.UTF8.GetString(piece));
    }

    static bool isprint(byte b) { return b >= 32 && b <= 126; }
    static bool isspace(byte b) { return b >= 9 && b <= 13; }

    static unsafe ulong random_u32(ulong* state) {
        unchecked {
            // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            return (*state * 0x2545F4914F6CDD1Dul) >> 32;
        }
    }

    static unsafe float random_f32(ulong* state) { // random float32 in [0,1)
        unchecked {
            return (random_u32(state) >> 8) / 16777216.0f;
        }
    }

    unsafe static int sample_mult(float* probabilities, int n, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    public static void Main(string[] args) {
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "D:\\llm.c\\gpt2_124M.bin");

        // build the Tokenizer
        Tokenizer tokenizer = new Tokenizer();
        tokenizer_init(tokenizer, "D:\\llm.c\\gpt2_tokenizer.bin");

        int B = 1; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int T = 32; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2

        // some memory for generating samples from the model
        ulong rng_state = 1337;
        int* gen_tokens = (int*)malloc(B * T, sizeof(int));
        int genT = T; // number of steps of inference we will do

        // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
        for (int i = 0; i < B * T; ++i) {
            gen_tokens[i] = 0;// tokenizer.eot_token;
        }
        // now sample from the model autoregressively
        for (int t = 1; t < genT; t++) {
            // note that inference is very wasteful here because for each token
            // we re-calculate the forward pass for all of (B,T) positions from scratch
            // but the inference here is just for sanity checking anyway
            // and we can maybe optimize a bit more later, with careful tests
            gpt2_forward(&model, gen_tokens, null, B, T);
            // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
            // we're in principle running B "inference streams" in parallel here
            // but only using position 0
            // get the Vp-dimensional vector probs[0, t-1, :]
            float* probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size;
            float coin = random_f32(&rng_state);
            // note we're only sampling from the first V elements, ignoring padding
            // (the probabilities in the padded region should be zero anyway)
            int next_token = sample_mult(probs, model.config.vocab_size, coin);
            gen_tokens[t] = next_token;
            // print the generated token, either using the Tokenizer or a fallback
            if (tokenizer.init_ok) {
                var token_str = tokenizer_decode(tokenizer, next_token);
                safe_printf(token_str);
            } else {
                // fall back to printing the token id
                printf("%d ", next_token);
            }
            // fflush(stdout);
        }
        printf("\n---\n");
    }
}