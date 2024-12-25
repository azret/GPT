using System;

namespace Microsoft.Win32 {
    public class WinUIModel : IDisposable, IWinUIModel {
        object _CriticalSection = new object();
        IntPtr[] _Views = null;
        public WinUIModel() {
        }
        ~WinUIModel() {
            Dispose(false);
        }
        public bool IsDisposed { get; internal set; }
        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing) {
            IsDisposed = true;
            if (disposing) {
                CloseWinUIClients();
            }
        }
        protected void PostWinUIMessage() {
            lock (_CriticalSection) {
                if (_Views != null) {
                    foreach (IntPtr hWnd in _Views) {
                        if (hWnd != IntPtr.Zero) {
                            User32.PostMessage(hWnd, WM.WINMM,
                                IntPtr.Zero,
                                IntPtr.Zero);
                        }
                    }
                }
            }
        }
        public void CloseWinUIClients() {
            lock (_CriticalSection) {
                if (_Views != null) {
                    foreach (IntPtr hWnd in _Views) {
                        if (hWnd != IntPtr.Zero) {
                            User32.PostMessage(hWnd, WM.CLOSE,
                                IntPtr.Zero,
                                IntPtr.Zero);
                        }
                    }
                }
            }
        }
        void IWinUIModel.AddWinUIClient(IntPtr hWnd) {
            if (hWnd == IntPtr.Zero) return;
            lock (_CriticalSection) {
                if (_Views == null) {
                    _Views = new IntPtr[0];
                }
                for (int i = 0; i < _Views.Length; i++) {
                    if (_Views[i] == hWnd) {
                        return;
                    }
                }
                for (int i = 0; i < _Views.Length; i++) {
                    if (_Views[i] == IntPtr.Zero) {
                        _Views[i] = hWnd;
                        return;
                    }
                }
                Array.Resize(ref _Views,
                    _Views.Length + 1);
                _Views[_Views.Length - 1] = hWnd;
            }
        }
        void IWinUIModel.RemoveWinUIClient(IntPtr hWnd) {
            if (hWnd == IntPtr.Zero) return;
            lock (_CriticalSection) {
                if (_Views != null) {
                    for (int i = 0; i < _Views.Length; i++) {
                        if (_Views[i] == hWnd) {
                            _Views[i] = IntPtr.Zero;
                        }
                    }
                }
            }
        }
    }
}