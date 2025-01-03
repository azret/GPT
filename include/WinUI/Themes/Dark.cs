﻿namespace System.Drawing {
    using System;
    using System.Threading;

#if NET5_0_OR_GREATER
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    public class Dark : ITheme {
        public readonly int ThreadId = Thread.CurrentThread.ManagedThreadId;
        class _Fonts {
            public readonly Font ExtraSmall = new Font("Consolas", 5.5f);
            public readonly Font Small = new Font("Consolas", 7.5f);
            public readonly Font Normal = new Font("Consolas", 11.5f);
            public readonly Font Large = new Font("Consolas", 14.5f);
            public readonly Font ExtraLarge = new Font("Consolas", 17.5f);
        }
        _Fonts Fonts = new _Fonts();
        public static class Colors {
            public static Color Background = Color.FromArgb(39, 39, 39);
            public static Color Foreground = Color.FromArgb(178, 178, 178);
            public static Color A = Color.FromArgb(255, 227, 158);
            public static Color B = Color.FromArgb(245, 87, 98);
            public static Color C = Color.FromArgb(112, 218, 255);
            public static Color D = Color.FromArgb(141, 227, 141);
            public static Color E = Color.White;
            public static Color LightLine = Color.FromArgb(66, 66, 66);
            public static Color DarkLine = Color.FromArgb(54, 54, 54);
            public static Color TitleBar = Color.FromArgb(32, 32, 32);
            public static Color TitleText = Color.FromArgb(0xFF, 0xFF, 0xFF);
            public static Color ChromeClose = Color.FromArgb(196, 43, 28);
            public static Color ChromeClosePressed = Color.FromArgb(181, 43, 30);
        }
        class _Brushes {
            public readonly Brush Background;
            public readonly Brush Foreground;
            public readonly Brush A;
            public readonly Brush B;
            public readonly Brush C;
            public readonly Brush D;
            public readonly Brush E;
            public readonly Brush TitleBar;
            public _Brushes() {
                Background = new SolidBrush(Colors.Background);
                Foreground = new SolidBrush(Colors.Foreground);
                A = new SolidBrush(Colors.A);
                B = new SolidBrush(Colors.B);
                C = new SolidBrush(Colors.C);
                D = new SolidBrush(Colors.D);
                E = new SolidBrush(Colors.E);
                TitleBar = new SolidBrush(Colors.TitleBar);
            }
        }
        _Brushes Brushes;
        class _Pens {
            public readonly Pen Background;
            public readonly Pen Foreground;
            public readonly Pen A;
            public readonly Pen B;
            public readonly Pen C;
            public readonly Pen D;
            public readonly Pen E;
            public readonly Pen LightLine;
            public readonly Pen DarkLine;
            public _Pens() {
                Background = new Pen(Colors.Background);
                Foreground = new Pen(Colors.Foreground);
                A = new Pen(Colors.A);
                B = new Pen(Colors.B);
                C = new Pen(Colors.C);
                D = new Pen(Colors.D);
                E = new Pen(Colors.E);
                LightLine = new Pen(Colors.LightLine);
                DarkLine = new Pen(Colors.DarkLine);
            }
        }
        _Pens Pens;
        public Dark() {
            Pens = new _Pens();
            Brushes = new _Brushes();
        }
        Font ITheme.GetFont(ThemeFont font) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (font) {
                case ThemeFont.ExtraSmall:
                    return Fonts.ExtraSmall;
                case ThemeFont.Small:
                    return Fonts.Small;
                case ThemeFont.Normal:
                    return Fonts.Normal;
                case ThemeFont.Large:
                    return Fonts.Large;
                case ThemeFont.ExtraLarge:
                    return Fonts.ExtraLarge;
            }
            throw new NotImplementedException();
        }
        Color ITheme.GetColor(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (color) {
                case ThemeColor.Background:
                    return Colors.Background;
                case ThemeColor.Foreground:
                    return Colors.Foreground;
                case ThemeColor.A:
                    return Colors.A;
                case ThemeColor.B:
                    return Colors.B;
                case ThemeColor.C:
                    return Colors.C;
                case ThemeColor.D:
                    return Colors.D;
                case ThemeColor.E:
                    return Colors.E;
                case ThemeColor.DarkLine:
                    return Colors.DarkLine;
                case ThemeColor.LightLine:
                    return Colors.LightLine;
                case ThemeColor.TitleBar:
                    return Colors.TitleBar;
                case ThemeColor.TitleText:
                    return Colors.TitleText;
                case ThemeColor.ChromeClose:
                    return Colors.ChromeClose;
                case ThemeColor.ChromeClosePressed:
                    return Colors.ChromeClosePressed;
            }
            throw new NotImplementedException();
        }
        Brush ITheme.GetBrush(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (color) {
                case ThemeColor.Background:
                    return Brushes.Background;
                case ThemeColor.Foreground:
                    return Brushes.Foreground;
                case ThemeColor.A:
                    return Brushes.A;
                case ThemeColor.B:
                    return Brushes.B;
                case ThemeColor.C:
                    return Brushes.C;
                case ThemeColor.D:
                    return Brushes.D;
                case ThemeColor.E:
                    return Brushes.E;
                case ThemeColor.TitleBar:
                    return Brushes.TitleBar;
            }
            throw new NotImplementedException();
        }
        Pen ITheme.GetPen(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (color) {
                case ThemeColor.Background:
                    return Pens.Background;
                case ThemeColor.Foreground:
                    return Pens.Foreground;
                case ThemeColor.A:
                    return Pens.A;
                case ThemeColor.B:
                    return Pens.B;
                case ThemeColor.C:
                    return Pens.C;
                case ThemeColor.D:
                    return Pens.D;
                case ThemeColor.E:
                    return Pens.E;
                case ThemeColor.LightLine:
                    return Pens.LightLine;
                case ThemeColor.DarkLine:
                    return Pens.DarkLine;
            }
            throw new NotImplementedException();
        }
    }
}