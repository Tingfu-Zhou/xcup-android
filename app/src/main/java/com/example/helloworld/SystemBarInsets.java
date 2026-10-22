package com.example.helloworld;

import android.app.Activity;
import android.os.Build;
import android.view.View;
import android.view.ViewGroup;

import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

/**
 * 系统栏避让。Android 15 起 target 35+ 的 app 被强制全屏绘制（edge-to-edge），target 36 起连退出的开关都没了：
 * 不处理的话，顶栏的返回键压在状态栏下面，底部内容压在导航栏下面（主题里的 statusBarColor 也不再生效）。
 * 这里按状态栏 / 导航栏 / 挖孔的 insets 把内容挪回安全区域。
 *
 * <p>Android 14 及以下窗口自己会避开系统栏，两个方法都什么也不做（避免和系统的留白叠加）。</p>
 */
final class SystemBarInsets {

    private static final int TYPES =
            WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout();

    private SystemBarInsets() {}

    /**
     * 普通页面：布局根 View 在原有 padding 上加系统栏的宽度。系统栏后面露出根 View 的背景色，
     * 和以前主题里 statusBarColor / navigationBarColor 设成 colorSurface 的样子一样。在 setContentView 之后调用。
     */
    static void padContentRoot(Activity activity) {
        if (Build.VERSION.SDK_INT < 35) return;
        ViewGroup content = activity.findViewById(android.R.id.content);
        if (content == null || content.getChildCount() == 0) return;
        View root = content.getChildAt(0);
        final int left = root.getPaddingLeft(), top = root.getPaddingTop();
        final int right = root.getPaddingRight(), bottom = root.getPaddingBottom();
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(TYPES);
            v.setPadding(left + bars.left, top + bars.top, right + bars.right, bottom + bars.bottom);
            return WindowInsetsCompat.CONSUMED;
        });
    }

    /**
     * 视频页：画面照样铺满整个屏幕（系统栏后面是黑底），只把浮在画面上的按钮在原有 margin 上挪进安全区域。
     * 全屏时系统栏隐藏，insets 只剩挖孔。overlays 必须是 root 里用 MarginLayoutParams 的子 View。
     */
    static void marginOverlays(View root, View... overlays) {
        if (Build.VERSION.SDK_INT < 35) return;
        final int[][] base = new int[overlays.length][];
        for (int k = 0; k < overlays.length; k++) {
            ViewGroup.MarginLayoutParams lp = (ViewGroup.MarginLayoutParams) overlays[k].getLayoutParams();
            base[k] = new int[]{lp.leftMargin, lp.topMargin, lp.rightMargin, lp.bottomMargin};
        }
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(TYPES);
            for (int k = 0; k < overlays.length; k++) {
                ViewGroup.MarginLayoutParams lp = (ViewGroup.MarginLayoutParams) overlays[k].getLayoutParams();
                lp.setMargins(base[k][0] + bars.left, base[k][1] + bars.top,
                        base[k][2] + bars.right, base[k][3] + bars.bottom);
                overlays[k].setLayoutParams(lp);
            }
            return insets;
        });
    }
}
