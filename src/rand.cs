#pragma warning disable CS8981

namespace torch.random {
    using System;

    public partial interface IRNG {
        uint randint32();
        float randfloat32();
        ulong randint64();
        double randfloat64();
    }

    public partial interface IRNG {
        internal void set_next_normal_sample(double? value);
        internal double? get_next_normal_sample();
    }

    public static unsafe partial class rand {
        public static void shuffle_(IRNG g, int[] data) {
            fixed (int* ptr = data) {
                shuffle_(g, ptr, data.Length);
            }
        }

        public static void shuffle_(IRNG g, int* data, int numel) {
            for (int i = 0; i < numel - 1; i++) {
                var z = g.randint32() % (numel - i);
                var t = data[i];
                data[i] = data[z + i];
                data[z + i] = t;
            }
        }

        /// <summary>
        /// Random permutation of integers from 0 to n - 1.
        /// </summary>
        public static void randperm_(IRNG g, int[] data) {
            fixed (int* ptr = data) {
                randperm_(g, ptr, data.Length);
            }
        }

        /// <summary>
        /// Random permutation of integers from 0 to n - 1.
        /// </summary>
        public static void randperm_(IRNG g, int* data, int numel) {
            for (int i = 0; i < numel; i++) {
                data[i] = i;
            }
            shuffle_(g, data, numel);
        }

        // Bernoulli Distribution

        public static void bernoulli_(IRNG g, float[] data, double p) {
            fixed (float* ptr = data) {
                bernoulli_(g, ptr, data.Length, p);
            }
        }

        public static void bernoulli_(IRNG g, float* data, int numel, double p) {
            if (data == null) throw new ArgumentNullException(nameof(data));
            var seed = g.randint32();
            var MCG31 = new mcg31m1(seed: seed);
            for (int i = 0; i < numel; i++) {
                if (MCG31.randfloat64() < p) {
                    data[i] = 1f;
                } else {
                    data[i] = 0f;
                }
            }
        }

        // Uniform Distribution

        public static float uniform(IRNG g, float from = 0, float to = 1) {
            return (float)g.randfloat32() * (to - from) + from; 
        }

        public static double uniform64(IRNG g, double from = 0, double to = 1) {
            return (double)g.randfloat64() * (to - from) + from;
        }

        public static void uniform_(IRNG g, float[] data, float from = 0, float to = 1) {
            fixed (float* ptr = data) {
                uniform_(g, ptr, data.Length, from, to);
            }
        }

        public static void uniform_(IRNG g, float* data, int numel, float from = 0, float to = 1) {
            for (int t = 0; t < numel; t++) {
                data[t] = uniform(g, from, to);
            }
        }

        // Normal Distribution

        public static void normal_(IRNG g, float[] data, float mean = 0, float std = 1) {
            fixed (float* ptr = data) {
                normal_(g, ptr, data.Length, mean, std);
            }
        }

        public static void normal_(IRNG g, float* data, int numel, float mean = 0, float std = 1) {
            // This implementation follows PyTorch so that we are numerically identical (as much as possible)
            //      when running verification tests.
            if (numel >= 16) {
                uniform_(g, data, numel);
                for (int i = 0; i < numel - 15; i += 16) {
                    fill16_(data + i, mean, std);
                }
                if (numel % 16 != 0) {
                    // recompute the last 16 values
                    data = data + numel - 16;
                    uniform_(g, data, 16);
                    fill16_(data, mean, std);
                }
                void fill16_(float* data, float mean, float std) {
                    for (int t = 0; t < 8; t++) {
                        var u1 = (1 - data[t]);
                        // for numerical stability if we draw a true 0 or a 1
                        if (u1 >= 1 - 1e-12f) {
                            u1 = 1 - 1e-12f;
                        }
                        if (u1 <= 1e-12f) {
                            u1 = 1e-12f;
                        }
                        var u2 = data[t + 8];
                        var radius = MathF.Sqrt(-2 * MathF.Log(u1));
                        var theta = 2.0f * MathF.PI * u2;
                        data[t] = (float)(radius * MathF.Cos(theta) * std + mean);
                        data[t + 8] = (float)(radius * MathF.Sin(theta) * std + mean);
                    }
                }
            } else {
                for (int t = 0; t < numel; t++) {
                    double? next_double_normal_sample = g.get_next_normal_sample();
                    if (next_double_normal_sample.HasValue) {
                        data[t] = (float)(next_double_normal_sample.Value * std + mean);
                        g.set_next_normal_sample(next_double_normal_sample = null);
                        continue;
                    }
                    // for numel < 16 we draw a double (float64)
                    double u1 = uniform64(g);
                    double u2 = uniform64(g);
                    // for numerical stability if we draw a true 0 or a 1
                    if (u2 >= 1 - 1e-12) {
                        u2 = 1 - 1e-12;
                    }
                    if (u2 <= 1e-12) {
                        u2 = 1e-12;
                    }
                    double radius = Math.Sqrt(-2 * Math.Log(1 - u2));
                    double theta = 2.0 * Math.PI * u1;
                    g.set_next_normal_sample(next_double_normal_sample = radius * Math.Sin(theta));
                    data[t] = (float)(radius * Math.Cos(theta) * std + mean);
                }
            }
        }
    }

    public static unsafe partial class rand {
        /// <summary>
        /// A Mersenne Twister pseudorandom number generator. Copyright(c) Makoto Matsumoto and Takuji Nishimura.
        /// </summary>
        public class mt19937 : IRNG {
            double? next_double_normal_sample = null;
            void IRNG.set_next_normal_sample(double? value) => next_double_normal_sample = value;
            double? IRNG.get_next_normal_sample() => next_double_normal_sample;
            static readonly uint[] MATRIX_A = { 0x0u, 0x9908b0df };
            const uint LMASK = 0x7fffffff;
            const uint UMASK = 0x80000000;
            const int MERSENNE_STATE_M = 397;
            const int MERSENNE_STATE_N = 624;
            uint[] state_;
            int left_;
            int next_;
            public mt19937(uint seed = 5489) {
                state_ = new uint[MERSENNE_STATE_N];
                left_ = 1;
                next_ = 0;
                init_with_uint32(seed);
            }
            void init_with_uint32(uint seed) {
                unchecked {
                    state_ = new uint[MERSENNE_STATE_N];
                    state_[0] = seed & 0xffffffff;
                    for (uint j = 1; j < MERSENNE_STATE_N; j++) {
                        state_[j] = 1812433253 * (state_[j - 1] ^ (state_[j - 1] >> 30)) + j;
                        state_[j] &= 0xffffffff;
                    }
                }
                left_ = 1;
                next_ = 0;
            }
            void next_state() {
                left_ = MERSENNE_STATE_N;
                next_ = 0;
                uint y, j;
                unchecked {
                    for (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
                        y = (state_[j] & UMASK) | (state_[j + 1] & LMASK);
                        state_[j] = state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ MATRIX_A[y & 0x1];
                    }
                    for (; j < MERSENNE_STATE_N - 1; j++) {
                        y = (state_[j] & UMASK) | (state_[j + 1] & LMASK);
                        state_[j] = state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ MATRIX_A[y & 0x1];
                    }
                    y = (state_[MERSENNE_STATE_N - 1] & UMASK) | (state_[0] & LMASK);
                    state_[MERSENNE_STATE_N - 1] = state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ MATRIX_A[y & 0x1];
                }
            }
            public uint randint32() {
                if (state_ == null) init_with_uint32(5489);
                if (--left_ <= 0) {
                    next_state();
                }
                uint y = state_[next_++];
                unchecked {
                    y ^= y >> 11;
                    y ^= (y << 7) & 0x9d2c5680;
                    y ^= (y << 15) & 0xefc60000;
                    y ^= y >> 18;
                }
                return y;
            }
            public ulong randint64() {
                return ((ulong)randint32() << 32) | randint32();
            }
            public float randfloat32() {
                return (randint32() & ((1ul << 24) - 1)) * (1.0f / (1ul << 24));
            }
            public double randfloat64() {
                return (randint64() & ((1ul << 53) - 1)) * (1.0d / (1ul << 53));
            }
        }

        /// <summary>
        /// The 31-bit multiplicative congruential pseudorandom number generator MCG(1132489760, 2^31 -1) [L'Ecuyer99]
        /// </summary>
        public class mcg31m1 {
            ulong state_;
            public mcg31m1(uint seed = 1) {
                state_ = seed % 0x000000007FFFFFFF;
            }

            public uint randint32() {
                uint x = (uint)(state_ % 0x000000007FFFFFFF);
                unchecked {
                    state_ = (1132489760 * state_) % 0x000000007FFFFFFF;
                }
                return x;
            }
            public ulong randint64() { return ((ulong)randint32() << 32) | randint32(); }
            public float randfloat32() { return (float)randint32() / 0x7FFFFFFF; }
            public double randfloat64() { return (double)randint32() / 0x7FFFFFFF; }
        }
    }
}
