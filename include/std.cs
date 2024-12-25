#pragma warning disable CS8981

using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using static kernel32;

public static partial class std {
    public static float[,] fromJagged(float[][] x, bool transpose = false) {
        int rows = x.Length;
        int cols = x.Where(k => k != null).Max(k => k.Length);
        var y = transpose ? new float[cols, rows] : new float[rows, cols];
        for (int n = 0; n < rows; n++) {
            for (int m = 0; m < cols; m++) {
                Assert<InvalidOperationException>.That(x[n].Length == cols);
                if (transpose) {
                    y[m, n] = x[n][m];
                } else {
                    y[n, m] = x[n][m];
                }
            }
        }
        return y;
    }

    public static double[,] toDouble(float[,] x) {
        var y = new double[x.GetLength(0), x.GetLength(1)];
        for (int row = 0; row < x.GetLength(0); row++) {
            for (int col = 0; col < x.GetLength(1); col++) {
                y[row, col] = x[row, col];
            }
        }
        return y;
    }

    public static float[] col(float[,] x, int col) {
        var y = new float[x.GetLength(0)];
        for (int row = 0; row < y.Length; row++) {
            y[row] = x[row, col];
        }
        return y;
    }


    public static float[] row(float[,] x, int row) {
        var y = new float[x.GetLength(1)];
        for (int col = 0; col < y.Length; col++) {
            y[col] = x[row, col];
        }
        return y;
    }


    public static float[] diag(float[,] x) {
        var dim = x.GetLength(0);
        var y = new float[x.GetLength(0)];
        for (int z = 0; z < dim; z++) {
            y[z] = x[z, z];
        }
        return y;
    }

    public static float[,] transpose(float[,] x) {
        var y = new float[x.GetLength(1), x.GetLength(0)];
        for (int i = 0; i < x.GetLength(0); i++) {
            for (int j = 0; j < x.GetLength(1); j++) {
                y[j, i] = x[i, j];
            }
        }
        return y;
    }

    // public static float[,] square(float[] x) {
    //     var dim = (int)MathF.Sqrt(x.Length);
    //     Assert<ArgumentOutOfRangeException>.That(dim * dim == x.Length, "Square matrix expected.");
    //     var y = new float[dim, dim];
    //     for (int n = 0, i = 0; i < dim; i++) {
    //         for (int j = 0; j < dim; j++) {
    //             y[i, j] = x[n++];
    //         }
    //     }
    //     return y;
    // }

    public static float[] flat(float[,] x) {
        var y = new float[x.Length];
        for (int i = 0, n = 0; i < x.GetLength(0); i++) {
            for (int j = 0; j < x.GetLength(1); j++) {
                y[n++] = x[i, j];
            }
        }
        return y;
    }

    public static float[] flat(float[][] x) {
        int len = 0;
        for (int i = 0; i < x.Length; i++) {
            len += x[i].Length;
        }
        var y = new float[len];
        for (int i = 0, n = 0; i < x.Length; i++) {
            for (int j = 0; j < x[i].Length; j++) {
                y[n++] = x[i][j];
            }
        }
        return y;
    }

    public static T[] shuffle_<T>(T[] A, ref ulong MCG) {
        if (MCG == 0) MCG = 13;
        ulong N = (ulong)A.Length;
        for (ulong i = 0; i < N - 1; i++) {
            // Inline MCG(1132489760, 2^31 -1) [L'Ecuyer99]
            ulong j = (MCG % 0x000000007FFFFFFF) % (ulong)A.Length;
            unchecked {
                MCG = 1132489760 * MCG;
                MCG = MCG % 0x000000007FFFFFFF;
            }
            if (j != i) {
                T t = A[i];
                A[i] = A[j];
                A[j] = t;
            }
        }
        return A;
    }

    public static void sort<T>(T[] A, int lo, int hi, Comparison<T> compare) {
        if (lo >= 0 && lo < hi) {
            int eq = lo, p = (lo + hi) / 2; int lt = lo, gt = hi;
            while (eq <= gt) {
                int c = compare(A[eq], A[p]);
                if (c < 0) {
                    T t = A[eq]; A[eq] = A[lt]; A[lt] = t;
                    lt++; eq++;
                } else if (c > 0) {
                    T t = A[eq]; A[eq] = A[gt]; A[gt] = t;
                    gt--;
                } else {
                    eq++;
                }
            }
            sort(A, lo, lt - 1, compare);
            sort(A, gt + 1, hi, compare);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float min(float x, float y) {
        return (float)MathF.Min(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong max(ulong x, ulong y) {
        return Math.Max(x, y);
    }

    public static void clampf_(float[] y, float min, float max) {
        for (int n = 0; n < y.Length; n++) {
            y[n] = Math.Clamp(y[n], min, max);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float maxf(float x, float y) {
        return MathF.Max(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sqrtf(float x) {
        return (float)MathF.Sqrt(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float rsqrtf(float x) {
        return (float)(1 / MathF.Sqrt(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float powf(float x, float y) {
        return (float)MathF.Pow(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float logf(float x) {
        return (float)MathF.Log(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float cosf(float x) {
        return (float)MathF.Cos(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sinf(float x) {
        return (float)MathF.Sin(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float expf(float x) {
        return (float)MathF.Exp(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float tanhf(float x) {
        return (float)MathF.Tanh(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float coshf(float x) {
        return (float)MathF.Cosh(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float fabsf(float x) {
        return (float)MathF.Abs(x);
    }

    public static float max(float[,] x) {
        var max = x[0, 0];
        for (int i = 0; i < x.GetLength(0); i++) {
            for (int j = 0; j < x.GetLength(1); j++) {
                if (x[i, j] > max) {
                    max = x[i, j];
                }
            }
        }
        return max;
    }

    public static unsafe float mean(float[] x) {
        var mean = 0f;
        for (int n = 0; n < x.Length; n++) {
            mean += x[n];
        }
        mean /= x.Length;
        return mean;
    }

    public static unsafe float mean(float* x, int numel) {
        var mean = 0f;
        for (int n = 0; n < numel; n++) {
            mean += x[n];
        }
        mean /= numel;
        return mean;
    }

    public static double mean(System.Collections.Generic.IEnumerable<double> x) {
        double mean = 0; ulong numel = 0;
        foreach (double d in x) {
            mean += d;
            numel += 1;
        }
        mean /= numel;
        return mean;
    }

    public static float mean(System.Collections.Generic.IEnumerable<float> x) {
        float mean = 0; ulong numel = 0;
        foreach (float f in x) {
            mean += f;
            numel += 1;
        }
        mean /= numel;
        return mean;
    }

    public static double stddev(System.Collections.Generic.IEnumerable<double> x, out double mu, out double var, ulong correction = 1) {
        mu = mean(x); ulong numel = 0;
        var = 0;
        foreach (double val in x) {
            var += Math.Pow(val - mu, 2);
            numel += 1;
        }
        var = var / (numel - correction);
        return Math.Sqrt(var);
    }

    /// <summary>
    /// Calculates the standard deviation with Bessel's correction.
    /// </summary>
    /// <param name="correction">Bessel's correction</param>
    public unsafe static float stddev(float[] x, out float mu, out float var, int correction = 1) {
        fixed (float* p = x) {
            return stddev(p, x.Length,
                out mu, out var, correction);
        }
    }

    public unsafe static float stddev(float* x, int numel, out float mu, out float var, int correction = 1) {
        mu = mean(x, numel);
        var = 0;
        for (int n = 0; n < numel; n++) {
            var += powf(x[n] - mu, 2);
        }
        var = var / (numel - correction);
        return sqrtf(var);
    }

    public unsafe static void normalize_(float[] y, int correction = 1) {
        fixed (float* p = y) {
            normalize_(p, y.Length, correction);
        }
    }

    public unsafe static void normalize_(float[,] y, int correction = 1) {
        fixed (float* p = y) {
            normalize_(p, y.Length, correction);
        }
    }

    public unsafe static void normalize_(float* y, int numel, int correction = 1) {
        var stddev = MathF.Max(
            std.stddev(y, numel,
                out var mu,
                out var var,
                     correction), 1e-12f);
        normalize_(y,
            numel,
                mu,
                stddev);
    }

    public unsafe static void normalize_(float[] y, out float mu, out float stddev) {
        fixed (float* p = y) {
            normalize_(p, y.Length,
                out mu, out stddev);
        }
    }
    public unsafe static void normalize_(float* y, int numel,
        out float mu, out float stddev, int correction = 1) {
        stddev = MathF.Max(
            std.stddev(y, numel,
                out mu,
                out var var,
                correction), 1e-12f);
        normalize_(y,
            numel,
            mu,
            stddev);
    }

    public unsafe static float[,] normalize(float[,] x, float mu, float stddev) {
        float[,] y = new float[x.GetLength(0), x.GetLength(1)];
        fixed (float* src = x) {
            fixed (float* dst = y) {
                CopyMemory(
                    dst,
                    src,
                    (ulong)y.Length * sizeof(float));
                normalize_(
                    dst,
                    y.Length,
                    mu,
                    stddev);
            }
        }
        return y;
    }

    public unsafe static void normalize_(float[,] y, float mu, float stddev) {
        fixed (float* p = y) {
            normalize_(
                p,
                y.Length,
                mu,
                stddev);
        }
    }

    public unsafe static void normalize_(float* y, int numel, float mu, float stddev) {
        for (int n = 0; n < numel; n++) {
            y[n] = (y[n] - mu) / stddev;
        }
    }

    public unsafe static void minmax(float* x, int numel, out float min, out float max) {
        min = x[0];
        max = x[0];
        for (int n = 1; n < numel; n++) {
            if (x[n] < min) min = x[n];
            if (x[n] > max) max = x[n];
        }
    }

    public unsafe static void minmax(float[] x, out float min, out float max) {
        fixed (float* p = x) {
            minmax(p, x.Length,
                out min, out max);
        }
    }

    public unsafe static void minmax_(float* y, int numel, out float min, out float max) {
        minmax(y,
            numel,
            out min,
            out max);
        max = MathF.Max(max, 1e-12f);
        normalize_(y,
            numel,
            min,
            max - min);
    }

    public unsafe static void minmax_(float[] y, out float min, out float max) {
        fixed (float* p = y) {
            minmax_(p, y.Length,
                out min, out max);
        }
    }

    public unsafe static void minmax_(float[,] y) {
        fixed (float* p = y) {
            minmax_(p, y.Length,
                out _, out _);
        }
    }

    public unsafe static void minmax_(float[,] y, out float min, out float max) {
        fixed (float* p = y) {
            minmax_(p, y.Length,
                out min, out max);
        }
    }

    public unsafe static void minmax_(float[] y) {
        fixed (float* p = y) {
            minmax_(p, y.Length,
                out _, out _);
        }
    }

    public static float trunc(float x, int digits) {
        var mult = (int)MathF.Pow(10, digits);
        var y = (MathF.Floor(x * mult) * (1.0 / mult));
        return (float)y;
    }

    public static unsafe float round(float x, int digits, MidpointRounding mode = MidpointRounding.ToEven) {
        var y = MathF.Round(x, digits, mode);
        return (float)y;
    }

    public static unsafe float[] round(float[] x, int digits = 0, MidpointRounding mode = MidpointRounding.ToEven) {
        float[] y = new float[x.Length];
        for (int n = 0; n < x.Length; n++) {
            y[n] = (float)MathF.Round(x[n], digits, mode);
        }
        return y;
    }

    public static unsafe float[] round(float* x, int numel, int digits, MidpointRounding mode = MidpointRounding.ToEven) {
        float[] y = new float[numel];
        for (int n = 0; n < numel; n++) {
            y[n] = (float)MathF.Round(x[n], digits, mode);
        }
        return y;
    }

    public static unsafe void round_(float[] y, int digits, MidpointRounding mode = MidpointRounding.ToEven) {
        fixed (float* p = y) {
            round_(p, y.Length, digits, mode);
        }
    }

    public static unsafe void round_(float* y, int numel, int digits, MidpointRounding mode = MidpointRounding.ToEven) {
        for (int n = 0; n < numel; n++) {
            y[n] = (float)MathF.Round(y[n], digits, mode);
        }
    }

    public static unsafe float[,] rect(float[] x, int rows) {
        int dim = x.Length / rows;
        if (dim * rows != x.Length) {
            throw new ArgumentException();
        }
        var y = new float[rows, dim];
        for (int i = 0, n = 0; i < rows; i++) {
            for (int j = 0; j < dim; j++) {
                y[i, j] = x[n++];
            }
        }
        return y;
    }


    public static unsafe float[,] square(float[] x) {
        fixed (float* p = x) {
            return square(p, x.Length, out _);
        }
    }

    public static unsafe float[,] square(float[] x, out int dim) {
        fixed (float* p = x) {
            return square(p, x.Length, out dim);
        }
    }

    public static unsafe float[,] square(float* x, int numel) {
        return square(x, numel, out _);
    }

    public static unsafe float[,] square(float* x, int numel, out int dim) {
         dim = (int)MathF.Sqrt(numel);
        if (dim * dim != numel) {
            throw new ArgumentException($"Square matrix cannot be created from {numel} elements.");
        }
        var y = new float[dim, dim];
        for (int row = 0, n = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
                y[row, col] = x[n++];
            }
        }
        return y;
    }

    public static unsafe float sum(float* x, int numel, bool NaN = true) {
        float y = 0;
        for (int n = 0; n < numel; n++) {
            if (NaN) {
                y += x[n];
            } else {
                if (float.IsNaN(x[n])) {
                    y += x[n];
                }
            }
        }
        return y;
    }

    public static unsafe float[] scal(float[] x, float scal = 1, float add = 0) {
        float[] y = new float[x.Length];
        for (int n = 0; n < x.Length; n++) {
            y[n] = (x[n] + add) * scal;
        }
        return y;
    }

    public static unsafe void scal_(float[] x, float scal = 1, float add = 0) {
        for (int n = 0; n < x.Length; n++) {
            x[n] = (x[n] + add) * scal;
        }
    }

    public static unsafe float[] scal(float* x, int numel, float scal = 1) {
        float[] y = new float[numel];
        for (int n = 0; n < numel; n++) {
            y[n] = x[n] * scal;
        }
        return y;
    }

    public static unsafe void scal_(float[,] y, float scal = 1) {
        fixed (float* p = y) {
            scal_(p, y.Length, scal);
        }
    }

    public static unsafe void scal_(float* y, int numel, float scal = 1) {
        for (int n = 0; n < numel; n++) {
            y[n] *= scal;
        }
    }

    public static unsafe int argmax(float[] x) {
        int y = 0;
        for (int n = 1; n < x.Length; n++) {
            if (x[n] > x[y]) {
                y = n;
            }
        }
        return y;
    }

    public static unsafe int argmax(float* x, int numel) {
        int y = 0;
        for (int n = 1; n < numel; n++) {
            if (x[n] > x[y]) {
                y = n;
            }
        }
        return y;
    }

    public static unsafe float[] expf(float* x, int numel) {
        float[] y = new float[numel];
        fixed (float* p = y) {
            expf_(p, numel);
        }
        return y;
    }

    public static unsafe void expf_(float* y, int numel) {
        for (int n = 0; n < numel; n++) {
            y[n] = expf(y[n]);
        }
    }

    public static unsafe void fill_(float* y, int numel, float fill) {
        for (int n = 0; n < numel; n++) {
            y[n] = fill;
        }
    }

    public static float[] ones(int numel, float scal = 1.0f) {
        float[] y = new float[numel];
        for (int n = 0; n < numel; n++) {
            y[n] = 1 * scal;
        }
        return y;
    }

    public static float[,] ones(int M, int N, float scal = 1.0f) {
        float[,] y = new float[M, N];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                y[m, n] = 1 * scal;
            }
        }
        return y;
    }

    public static unsafe void log_(float[] y, float scal = 1.0f, float add = 1e-12f) {
        fixed (float* p = y) {
            log_(p, y.Length, scal, add);
        }
    }

    public static unsafe void log_(float[,] y, float scal = 1.0f, float add = 1e-12f) {
        fixed (float* p = y) {
            log_(p, y.Length, scal, add);
        }
    }

    public static unsafe void log_(float* y, int numel, float scal = 1.0f, float add = 1e-12f) {
        for (int n = 0; n < numel; n++) {
            y[n] = logf(y[n] + add) * scal;
        }
    }

    public static unsafe void exp_(float[,] y) {
        fixed (float* p = y) {
            exp_(p, y.Length);
        }
    }

    public static unsafe void exp_(float* y, int numel) {
        for (int n = 0; n < numel; n++) {
            y[n] = expf(y[n]);
        }
    }

    public static unsafe void ones_(float* y, int numel, float scal = 1.0f) {
        for (int n = 0; n < numel; n++) {
            y[n] = 1 * scal;
        }
    }

    public static float[] zeros(int numel) {
        float[] y = new float[numel];
        for (int n = 0; n < numel; n++) {
            y[n] = 0;
        }
        return y;
    }

    public static unsafe void zeros_(float* y, int numel) {
        for (int n = 0; n < numel; n++) {
            y[n] = 0;
        }
    }

    public static float pdf(float x, float mu, float stddev) {
        var ω = (x - mu) / stddev;
        return (float)(MathF.Exp(-0.5f * ω * ω)
            / (2.5066282746310005024157652848110452530069867406099d * stddev));
    }

    static unsafe ulong xorshift32(ulong* state) {
        /* See href="https://en.wikipedia.org/wiki/Xorshift#xorshift.2A" */
        unchecked {
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            return (*state * 0x2545F4914F6CDD1Dul) >> 32;
        }
    }

    /// <summary>
    /// (-2147483648, +2147483647)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int rand(ulong* state) {
        unchecked {
            return (int)(xorshift32(state));
        }
    }

    /// <summary>
    /// (0, +4294967295)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe uint urand(ulong* state) {
        unchecked {
            return (uint)(xorshift32(state));
        }
    }

    /// <summary>
    /// (-1, +1)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float randf(ulong* state) {
        unchecked {
            return (int)xorshift32(state) / 2147483647.0f;
        }
    }

    /// <summary>
    /// (0, +1)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float urandf(ulong* state) {
        unchecked {
            return (uint)xorshift32(state) / 4294967295.0f;
        }
    }

    public static unsafe float[] randf(int N, ulong* state) {
        float[] Y = new float[N];
        for (int i = 0; i < N; i++) {
            Y[i] = randf(state);
        }
        return Y;
    }

    /// <summary>
    /// (-6.2831853071795862, +6.2831853071795862)
    /// </summary>
    public static unsafe float normal(ulong* state, float mean = 0, float std = 1) {
        // Box–Muller transform
        return (float)((std * MathF.Sqrt(-2.0f * MathF.Log(urandf(state) + 1e-12f))) * MathF.Sin(2 * 3.1415926535897931f * urandf(state)) + mean);
    }


    public static void fclose(IntPtr hFile) {
        if (hFile != IntPtr.Zero && hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
    }

    public static IntPtr fopen(string fileName, string mode = "rb") {
        CreationDisposition nCreationDisposition = CreationDisposition.OpenExisting;
        uint dwDesiredAccess = GENERIC_READ;
        switch (mode) {
            case "r":
            case "rb":
                break;
            case "r+":
            case "r+b":
            case "rb+":
                dwDesiredAccess = GENERIC_WRITE;
                nCreationDisposition = CreationDisposition.OpenExisting;
                break;
            case "w":
            case "wb":
                dwDesiredAccess = GENERIC_WRITE;
                nCreationDisposition = CreationDisposition.CreateAlways;
                break;
            default:
                throw new NotSupportedException($"The specified mode '{mode}' is not supported.");
        }
        var hFile = CreateFile(Path.GetFullPath(fileName),
                     dwDesiredAccess,
                     ShareMode.Read,
                     IntPtr.Zero,
                     nCreationDisposition,
                     FILE_ATTRIBUTE_NORMAL,
                     IntPtr.Zero);
        if (hFile == INVALID_HANDLE_VALUE) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            const int ERROR_FILE_NOT_FOUND = 2;
            if (ERROR_FILE_NOT_FOUND == lastWin32Error) {
                throw new FileNotFoundException("File not found.", fileName);
            }
            throw new Win32Exception(lastWin32Error);
        }
        return hFile;
    }

    public unsafe static uint fwrite(byte val, IntPtr hFile) { return fwrite(&val, sizeof(byte), 1, hFile); }
    public unsafe static uint fwrite(byte[] _Buffer, int count, IntPtr hFile) { fixed (void* ptr = _Buffer) { return fwrite(ptr, sizeof(byte), count, hFile); } }
    public unsafe static uint fwrite(float[] _Buffer, int count, IntPtr hFile) { fixed (void* ptr = _Buffer) { return fwrite(ptr, sizeof(float), count, hFile); } }
    public unsafe static uint fwrite(void* _Buffer, int _ElementSize, int _ElementCount, IntPtr hFile) {
        uint nNumberOfBytesToWrite = checked((uint)_ElementSize * (uint)_ElementCount);
        if (nNumberOfBytesToWrite == 0) {
            return 0;
        }
        int bResult = WriteFile(
            hFile,
            _Buffer,
            nNumberOfBytesToWrite,
            out uint numberOfBytesWritten,
            IntPtr.Zero);
        if (bResult == 0) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            if (lastWin32Error == 232) {
                return 0;
            }
            throw new Win32Exception(lastWin32Error);
        }
        return numberOfBytesWritten;
    }

    public unsafe static uint fread(byte[] _Buffer, int count, IntPtr hFile) {
        fixed (void* ptr = _Buffer) {
            return fread(ptr, sizeof(byte), count, hFile);
        }
    }

    public unsafe static uint fread(void* _Buffer, int _ElementSize, int _ElementCount, IntPtr hFile) {
        uint nNumberOfBytesToRead = checked((uint)_ElementSize * (uint)_ElementCount);
        if (nNumberOfBytesToRead == 0) {
            return 0;
        }
        const int ERROR_BROKEN_PIPE = 109;
        int bResult = ReadFile(
            hFile,
            _Buffer,
            nNumberOfBytesToRead,
            out uint numberOfBytesRead,
            IntPtr.Zero);
        if (bResult == 0) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            if (lastWin32Error == ERROR_BROKEN_PIPE) {
                return 0;
            }
            throw new Win32Exception(lastWin32Error);
        }
        return numberOfBytesRead;
    }

    public unsafe static ulong fseek(IntPtr hFile, long offset, SeekOrigin origin) {
        int lastWin32Error;
        int lo = (int)offset,
           hi = (int)(offset >> 32);
        lo = SetFilePointerWin32(hFile, lo, &hi, (int)origin);
        if (lo == -1 && (lastWin32Error = Marshal.GetLastWin32Error()) != 0) {
            throw new Win32Exception(lastWin32Error);
        }
        return (((ulong)(uint)hi << 32) | (uint)lo);
    }

    public unsafe static long ftell(IntPtr hFile) {
        int lastWin32Error;
        int hi = 0;
        int lo = SetFilePointerWin32(hFile, 0, &hi, (int)SeekOrigin.Current);
        if (lo == -1 && (lastWin32Error = Marshal.GetLastWin32Error()) != 0) {
            throw new Win32Exception(lastWin32Error);
        }
        return (((long)((uint)hi << 32) | (uint)lo));
    }

    public static ulong fsize(IntPtr hFile) {
        int lowSize = GetFileSize(hFile, out int highSize);
        if (lowSize == -1) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            throw new Win32Exception(lastWin32Error);
        }
        return ((ulong)highSize << 32) | (uint)lowSize;
    }

    public static unsafe void* malloc(int _ElementCount, int _ElementSize) {
        return malloc(checked((ulong)_ElementCount * (ulong)_ElementSize));
    }

    public static unsafe void* malloc(ulong size) {
        if (size <= 0) throw new ArgumentOutOfRangeException("size");
        var hglobal = LocalAlloc(LMEM_FIXED, size);
        if (hglobal == null) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        return hglobal;
    }

    public static unsafe void* realloc(void* hglobal, ulong size) {
        var hglobal_ = LocalReAlloc(hglobal, size, LMEM_MOVEABLE);
        if (hglobal_ == null) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        return hglobal_;
    }

    public static unsafe void free(void* hglobal) {
        if (hglobal != null)
            LocalFree(hglobal);
    }

    public static unsafe void free(ref IntPtr hglobal) {
        if (hglobal != IntPtr.Zero)
            LocalFree(hglobal);
        hglobal = IntPtr.Zero;
    }

    public static string memsize(ulong size) {
        string[] sizes = { "B", "KB", "MB", "GB", "TB" };
        double len = size;
        int order = 0;
        while (len >= 1024 && order < sizes.Length - 1) {
            order++;
            len = len / 1024;
        }
        return string.Format("{0:f4} {1}", len, sizes[order]);
    }
    public static string size(ulong size) {
        // thousand million billion trillion
        string[] sizes = { "", "K", " M", " B", " T" };
        double len = size;
        int order = 0;
        while (len >= 1000 && order < sizes.Length - 1) {
            order++;
            len = len / 1000;
        }
        return string.Format("{0:f1}{1}", len, sizes[order]);
    }

    public static void printf(string fmt, params object[] args) {
        fprintf(Console.Out, fmt, args);
    }

    public static void fprintf(TextWriter _Stream, string fmt, params object[] args) {
        for (int i = 0; i < args.Length; i++) {
            var pos = fmt.IndexOf("%");
            if (pos < 0 || pos + 1 >= fmt.Length) {
                throw new ArgumentOutOfRangeException();
            }
            string s = fmt.Substring(
                0,
                pos);
            int skip = 2;
            switch (fmt[pos + 1]) {
                case 'f':
                    if (pos + 2 < fmt.Length && char.IsDigit(fmt[pos + 2])) {
                        s += "{" + i.ToString() + ":F" + fmt[pos + 2] + "}";
                        skip++;
                    } else {
                        s += "{" + i.ToString() + ":F6}";
                    }
                    break;
                case 'x':
                    if (pos + 2 < fmt.Length && char.IsDigit(fmt[pos + 2])) {
                        s += "{" + i.ToString() + ":x" + fmt[pos + 2] + "}";
                        skip++;
                    } else {
                        s += "{" + i.ToString() + ":x}";
                    }
                    break;
                case 'z':
                    s += "{" + i.ToString() + "}";
                    if (pos + 2 < fmt.Length && fmt[pos + 2] == 'u') {
                        skip++;
                    }
                    break;
                case 'l':
                    s += "{" + i.ToString() + "}";
                    if (pos + 2 < fmt.Length && fmt[pos + 2] == 'l') {
                        skip++;
                    }
                    if (pos + 3 < fmt.Length && (fmt[pos + 3] == 'd' || fmt[pos + 3] == 'u')) {
                        skip++;
                    }
                    break;
                case 'd':
                case 's':
                case 'g':
                case 'e':
                    s += "{" + i.ToString() + "}";
                    break;
                default:
                    throw new NotImplementedException();
            }
            s += fmt.Substring(
                pos + skip);
            fmt = s;
        }
        _Stream.Write(fmt, args);
    }
}