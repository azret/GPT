﻿namespace System.Drawing {
    using System;
    using System.Threading;

#if NET5_0_OR_GREATER
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    public class Light : ITheme {
        public readonly int ThreadId = Thread.CurrentThread.ManagedThreadId;
        class _Fonts {
            public readonly Font ExtraSmall = new Font("Consolas", 5.5f);
            public readonly Font Small = new Font("Consolas", 7.5f);
            public readonly Font Normal = new Font("Consolas", 11.5f);
            public readonly Font Large = new Font("Consolas", 14.5f);
            public readonly Font ExtraLarge = new Font("Consolas", 17.5f);
        }
        _Fonts Fonts = new _Fonts();
        class _Colors {
            public readonly Color Background = Color.FromArgb(39, 39, 39);
            public readonly Color Foreground = Color.FromArgb(178, 178, 178);
            public readonly Color A = Color.FromArgb(255, 227, 158);
            public readonly Color B = Color.FromArgb(245, 87, 98);
            public readonly Color C = Color.FromArgb(112, 218, 255);
            public readonly Color D = Color.FromArgb(141, 227, 141);
            public readonly Color E = Color.White;
            public readonly Color LightLine = Color.FromArgb(66, 66, 66);
            public readonly Color DarkLine = Color.FromArgb(54, 54, 54);
            public readonly Color TitleBar = Color.FromArgb(32, 32, 32);
            public readonly Color TitleText = Color.FromArgb(0xFF, 0xFF, 0xFF);
            public readonly Color ChromeClose = Color.FromArgb(196, 43, 28);
            public readonly Color ChromeClosePressed = Color.FromArgb(181, 43, 30);
        }
        _Colors Colors = new _Colors();
        class _Brushes {
            _Colors Colors;
            public readonly Brush Background;
            public readonly Brush Foreground;
            public readonly Brush A;
            public readonly Brush B;
            public readonly Brush C;
            public readonly Brush D;
            public readonly Brush E;
            public readonly Brush TitleBar;
            public _Brushes(_Colors colors) {
                Colors = colors;
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
            _Colors Colors;
            public readonly Pen Background;
            public readonly Pen Foreground;
            public readonly Pen A;
            public readonly Pen B;
            public readonly Pen C;
            public readonly Pen D;
            public readonly Pen E;
            public readonly Pen LightLine;
            public readonly Pen DarkLine;
            public _Pens(_Colors colors) {
                Colors = colors;
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

        Color[] __Color = new Color[(int)ThemeColor.Last];

        public Light() {
            __Color[(int)ThemeColor.Background] = Color.White;
            __Color[(int)ThemeColor.Foreground] = Color.FromArgb(0xFFFFF);
            __Color[(int)ThemeColor.A] = Color.FromArgb(255, 227, 158);
            __Color[(int)ThemeColor.B] = Color.FromArgb(245, 87, 98);
            __Color[(int)ThemeColor.C] = Color.FromArgb(112, 218, 255);
            __Color[(int)ThemeColor.D] = Color.FromArgb(141, 227, 141);
            __Color[(int)ThemeColor.E] = Color.Black;
            __Color[(int)ThemeColor.DarkLine] = Color.FromArgb(0xFFFFF);
            __Color[(int)ThemeColor.LightLine] = Color.FromArgb(0xFFFFF);
            __Color[(int)ThemeColor.TitleBar] = Color.FromArgb(0xF3F3F3);
            __Color[(int)ThemeColor.TitleText] = Color.Black;
            __Color[(int)ThemeColor.ChromeClose] = Color.FromArgb(196, 43, 28);
            __Color[(int)ThemeColor.ChromeClosePressed] = Color.FromArgb(181, 43, 30);
        }

        public Color GetColor(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            if ((int)color >= 0 && (int)color < __Color.Length) {
                return __Color[(int)color];
            }
            throw new ArgumentOutOfRangeException();
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

        Brush[] __Brush;

        Brush ITheme.GetBrush(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            if (__Brush is null) {
                __Brush = new Brush[__Color.Length];
            }
            if ((int)color >= 0 && (int)color < __Brush.Length) {
                if (__Brush[(int)color] is null) {
                    __Brush[(int)color] = new SolidBrush(GetColor(color));
                }
                return __Brush[(int)color];
            }
            throw new ArgumentOutOfRangeException();
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