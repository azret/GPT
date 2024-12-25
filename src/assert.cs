using System.Diagnostics;

namespace System {
    public static class Assert<T> where T : Exception, new () {
#nullable enable
        [DebuggerHidden]
        [DebuggerStepThrough]
        public static void That(bool condition, params object?[]? kwargs) {
            if (kwargs == null || kwargs.Length == 0) {
                if (!condition) {
                    T ex = new T();
                    throw ex;
                }
            } else {
                if (!condition) {
                    T ex = (T?)Activator.CreateInstance(typeof(T), kwargs) ?? new T();
                    throw ex;
                }
            }
        }
#nullable disable
    }
}