using System;
using System.Drawing;
using System.Runtime.InteropServices;

namespace Microsoft.Win32.Plots {
    public unsafe class HeatmapWinUIViewModel : WinUIModel {
        Array _array;
        float* _data;
        int _numel;
        GCHandle _pin;
        public HeatmapWinUIViewModel(IntPtr data, int numel) {
            _data = (float*)data;
            _numel = numel;
        }
        public static explicit operator HeatmapWinUIViewModel(float[,] data) {
            return new HeatmapWinUIViewModel(data);
        }
        public static explicit operator HeatmapWinUIViewModel(float[] data) {
            return new HeatmapWinUIViewModel(data);
        }
        public HeatmapWinUIViewModel(float[,] data) {
            _array = (float[,])data.Clone();
            _pin = GCHandle.Alloc(_array, GCHandleType.Pinned);
            _data = (float*)_pin.AddrOfPinnedObject();
            _numel = _array.Length;
        }
        public HeatmapWinUIViewModel(float[] data) {
            _array = (float[])data.Clone();
            _pin = GCHandle.Alloc(_array, GCHandleType.Pinned);
            _data = (float*)_pin.AddrOfPinnedObject();
            _numel = _array.Length;
        }
        public unsafe int numel => _numel;
        public unsafe float* data => _data;
        protected override void Dispose(bool disposing) {
            _numel = 0;
            _data = null;
            if (_pin.IsAllocated) {
                _pin.Free();
            }
            _array = null;
            base.Dispose(disposing);
        }
    }

#if NET5_0_OR_GREATER
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    public unsafe class HeatmapWinUI2 : WinUIController<HeatmapWinUIViewModel> {
        public static void Show(string text, float[] data) {
            var dim = (int)Math.Sqrt(data.Length);
            if (dim * dim != data.Length) {
                throw new ArgumentException("Square matrix expected.");
            }
            WinUI.StartWinUI(text, new HeatmapWinUI2(
                dim, dim, (HeatmapWinUIViewModel)data));
        }
        public static void Show(string text, float[,] data) {
            WinUI.StartWinUI(text, new HeatmapWinUI2(
                data.GetLength(0),
                data.GetLength(1), (HeatmapWinUIViewModel)data));
        }
        public override int WindowWidth { get; set; } = 720;
        public override int WindowHeight { get; set; } = 720;
        readonly int Rows;
        readonly int Cols;
        public HeatmapWinUI2(int rows, int cols, HeatmapWinUIViewModel model)
            : base(model,
                  WinUIControllerOptions.MuteOnSpaceBar |
                  WinUIControllerOptions.UnMuteModelOnOpen |
                  WinUIControllerOptions.DisposeModelOnClose) {
            ArgumentOutOfRangeException.ThrowIfLessThan(Model.numel, checked(cols * rows));
            Rows = rows;
            Cols = cols;
        }

        public override void OnPaint(IWinUI winUI, Graphics g, RectangleF r) {
            float[] data = new float[Model.numel];
            fixed (float* dst = data) {
                kernel32.CopyMemory(
                    dst,
                    Model.data,
                    (ulong)Model.numel * sizeof(float));
            }

            var mean = std.mean(data);

            float[] norm = (float[])data.Clone();
            std.log_(
                norm);
            std.normalize_(
                norm);

            // std.log_(norm);

            float H = r.Height / Math.Min(Rows, 32);
            float W = r.Width / Math.Min(Cols, 32);
            for (int y = 0; y < Math.Min(Rows, 32); y++) {
                for (int x = 0; x < Math.Min(Cols, 32); x++) {
                    var square = new RectangleF(
                        x * W,
                        y * H,
                        W,
                        H);
                    Color bgColor;
                    var originalValue = data[y * Cols + x];
                    var normalizedValue = norm[y * Cols + x];
                    if (normalizedValue >= 1) {
                        bgColor = Color.DarkRed;
                    } else if (normalizedValue >= 0 && normalizedValue < 1) {
                        bgColor = Gdi.Blend(Color.DarkRed, Color.White, normalizedValue);
                    } else if (normalizedValue < 0 && normalizedValue > -1) {
                        bgColor = Gdi.Blend(Color.Black, Color.White, -normalizedValue);
                    } else {
                        bgColor = Color.Black;
                    }
                    var invertedColor = Color.FromArgb(255 - bgColor.R, 255 - bgColor.G, 255 - bgColor.B);
                    var bg = new SolidBrush(bgColor);
                    g.FillRectangle(
                        bg,
                        square);
                    if (true) {
                        var fg = new SolidBrush(invertedColor);
                        var font = new Font("Consolas", Math.Min(Math.Max(H / 7, 6.5f), 11));
                        var pt = new PointF(square.X + 4, square.Y + 4);
                        var s = $"{originalValue:f2}";
                        var sz = g.MeasureString(s, font);
                        g.DrawString(
                            s,
                            font,
                            fg,
                            square.X + square.Width / 2 - sz.Width / 2,
                            square.Y + square.Height / 2 - sz.Height / 2);
                        font.Dispose();
                        fg.Dispose();
                        // var mfg = new SolidBrush(Color.Green);
                        // var mfont = new Font("Segoe UI Symbol", Math.Min(Math.Max(H / 6, 11), 13));
                        // var ms = $"🔼";
                        // var msz = g.MeasureString(ms, mfont);
                        // g.DrawString(
                        //     ms,
                        //     mfont,
                        //     mfg,
                        //     square.X,
                        //     square.Y + square.Height / 2 - msz.Height / 2);
                        // mfont.Dispose();
                        // mfg.Dispose();
                    }
                    bg.Dispose();
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    public unsafe class SignalWinUIController : WinUIController<float[][]> {
         public override int WindowWidth { get; set; } = 720;
        public override int WindowHeight { get; set; } = 720;
        public SignalWinUIController(params float[][] data)
            : base(data,
                  WinUIControllerOptions.MuteOnSpaceBar |
                  WinUIControllerOptions.UnMuteModelOnOpen |
                  ~WinUIControllerOptions.DisposeModelOnClose) {
        }
        public override void OnPaint(IWinUI winUI, Graphics g, RectangleF r) {
            var min = float.PositiveInfinity; var max = float.NegativeInfinity;
            for (int m = 0; m < Model.Length; m++) {
                var s = Model[m];
                if (s == null) continue;
                foreach (float f in s) {
                    if (!float.IsNaN(f)) {
                        if (f < min) {
                            min = f;
                        }
                        if (f > max) {
                            max = f;
                        }
                    }
                }
            }
            float a = -0.4f;
            float b = 1;
            Gdi.DrawPaper(g, r, winUI.Theme);
            var c = ThemeColor.A;
            for (int m = 0; m < Model.Length; m++) {
                var s = Model[m];
                if (s == null) continue;
                Gdi.DrawSignal(g,
                    r,
                    winUI.Theme.GetBrush(c),
                    (n) => a + (s[n] - min) / (max - min) * (b - a),
                    SignalType.Curve,
                    s.Length,
                    1.7f);
                c = c.Next();
            }
        }
    }

    public unsafe class Heatmap : WinUIController<float[,]> {
        static Color RGB(byte r, byte g, byte b) {
            return Color.FromArgb(r, g, b);
        }
        static int[] Colors = [
            -10026977,
            -9830113,
            -9633248,
            -9436384,
            -9239519,
            -9042655,
            -8845790,
            -8648926,
            -8452061,
            -8320989,
            -8124124,
            -7927260,
            -7730395,
            -7533531,
            -7336666,
            -7139802,
            -6942937,
            -6746073,
            -6549209,
            -6352344,
            -6155480,
            -5958615,
            -5761751,
            -5564886,
            -5368022,
            -5171157,
            -5039828,
            -4973523,
            -4841682,
            -4775376,
            -4709071,
            -4577230,
            -4511180,
            -4379339,
            -4313034,
            -4246728,
            -4114887,
            -4048838,
            -3916996,
            -3850691,
            -3784386,
            -3652544,
            -3586239,
            -3454654,
            -3388348,
            -3256507,
            -3190202,
            -3123896,
            -2992055,
            -2926006,
            -2794164,
            -2727859,
            -2661553,
            -2595503,
            -2463661,
            -2397355,
            -2331049,
            -2264999,
            -2198692,
            -2132386,
            -2000800,
            -1934494,
            -1868188,
            -1802138,
            -1735832,
            -1669526,
            -1537684,
            -1471634,
            -1405328,
            -1339022,
            -1272972,
            -1141129,
            -1074823,
            -1008517,
            -942467,
            -876161,
            -809855,
            -743805,
            -677754,
            -677239,
            -676725,
            -610418,
            -609903,
            -609388,
            -543338,
            -542823,
            -542308,
            -476258,
            -475743,
            -475228,
            -408921,
            -408407,
            -407892,
            -341841,
            -341327,
            -275276,
            -274761,
            -274247,
            -207940,
            -207425,
            -206910,
            -140860,
            -140345,
            -140087,
            -139829,
            -205107,
            -204849,
            -204592,
            -204078,
            -269356,
            -269098,
            -268840,
            -268582,
            -333860,
            -333602,
            -333345,
            -333087,
            -398365,
            -397851,
            -397593,
            -397335,
            -397077,
            -462355,
            -462097,
            -461840,
            -461582,
            -526860,
            -526602,
            -591881,
            -657673,
            -789002,
            -854538,
            -985866,
            -1051659,
            -1182987,
            -1248523,
            -1379851,
            -1445644,
            -1576716,
            -1642508,
            -1773836,
            -1839629,
            -1970701,
            -2036493,
            -2167822,
            -2233358,
            -2364686,
            -2430478,
            -2561551,
            -2627343,
            -2758671,
            -2824463,
            -2955536,
            -3021328,
            -3152657,
            -3349777,
            -3481106,
            -3677971,
            -3809300,
            -4006420,
            -4137749,
            -4334614,
            -4465942,
            -4663063,
            -4794392,
            -4991256,
            -5122585,
            -5319706,
            -5451035,
            -5647899,
            -5779228,
            -5910813,
            -6107677,
            -6239006,
            -6435871,
            -6567456,
            -6764320,
            -6895649,
            -7092514,
            -7289635,
            -7486756,
            -7683877,
            -7880998,
            -8078119,
            -8275240,
            -8472361,
            -8669482,
            -8866603,
            -9063724,
            -9326381,
            -9523502,
            -9720623,
            -9917488,
            -10114609,
            -10311730,
            -10508851,
            -10705973,
            -10903094,
            -11100215,
            -11362872,
            -11559993,
            -11757114,
            -11954235,
            -12151356,
            -12348477,
            -12414526,
            -12546111,
            -12611904,
            -12677953,
            -12809538,
            -12875586,
            -12941379,
            -13072964,
            -13139013,
            -13205062,
            -13336391,
            -13402440,
            -13468489,
            -13600074,
            -13665867,
            -13731915,
            -13863500,
            -13929549,
            -13995598,
            -14126927,
            -14192976,
            -14259025,
            -14390610,
            -14456403,
            -14522452,
            -14654037,
            -14720088,
            -14786139,
            -14852190,
            -14918497,
            -14984548,
            -15050599,
            -15116650,
            -15182701,
            -15248752,
            -15380339,
            -15446390,
            -15512441,
            -15578748,
            -15644799,
            -15710850,
            -15776901,
            -15842951,
            -15909002,
            -15975053,
            -16106640,
            -16172947,
            -16238998,
            -16305049,
            -16371100,
            -16437151,
            ];
        public static void Show(string text, float[,] data) {
            WinUI.StartWinUI(text, new Heatmap(data));
        }
        public override int WindowWidth { get; set; } = 720;
        public override int WindowHeight { get; set; } = 720;
        public Heatmap(float[,] data)
            : base((float[,])data.Clone(),
                  WinUIControllerOptions.MuteOnSpaceBar |
                  WinUIControllerOptions.UnMuteModelOnOpen |
                  WinUIControllerOptions.DisposeModelOnClose) {
        }
        public override void OnPaint(IWinUI winUI, Graphics g, RectangleF r) {
            float[,] data = Model;
            float[,] norm = std.normalize(data, 0f, float.E);

            float H = r.Height / data.GetLength(0);
            float W = r.Width / data.GetLength(1);

            for (int y = 0; y < data.GetLength(0); y++) {

                for (int x = 0; x < data.GetLength(1); x++) {
                    var square = new RectangleF(
                        x * W,
                        y * H,
                        W,
                        H);

                    var value = data[y, x];
                    var normalizedValue = norm[y, x];

                    Color bgColor;

                    float Map(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
                        return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
                    }

                    if (float.IsNaN(value)) {
                        bgColor = Color.Black;
                    } else {
                        if (normalizedValue >= 0) {
                            var amp = (1 - Math.Min(Math.Abs(normalizedValue), 1));
                            amp = Map(amp, 0, 1, 0, Colors.Length / 2 - 1);
                            bgColor = Color.FromArgb(Colors[(int)amp]);
                        } else {
                            var amp = (Math.Min(Math.Abs(normalizedValue), 1));
                            amp = Map(amp, 0, 1, Colors.Length / 2, Colors.Length - 1);
                            bgColor = Color.FromArgb(Colors[(int)amp]);
                        }
                    }

                    var invertedColor = Color.FromArgb(255 - bgColor.R, 255 - bgColor.G, 255 - bgColor.B);

                    var bg = new SolidBrush(bgColor);
                    g.FillRectangle(
                        bg,
                        square);
                    if (true) {
                        //-----------------//
                        var fs = FontStyle.Regular;
                        var fc = invertedColor;
                        var fz = Math.Min(Math.Max(H / 8, 6.3f), 8.3f);
                        var fg = new SolidBrush(fc);
                        var font = new Font("Consolas", fz, fs);
                        var pt = new PointF(square.X + 4, square.Y + 4);
                        var s = $"{value:f2}";
                        var sz = g.MeasureString(s, font);
                        g.DrawString(
                            s,
                            font,
                            fg,
                            square.X + square.Width / 2 - sz.Width / 2,
                            square.Y + square.Height / 2 - sz.Height / 2);
                        fg.Dispose();
                        font.Dispose();
                    }
                    bg.Dispose();
                }
            }
        }
    }
}