// NEW file: UpdateChecker.java
package com.example.helloworld;

import android.content.Context;
import android.os.Build;


import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * 极简版本检查器：
 * - 拉取远端 version.json
 * - 对比 BuildConfig.VERSION_CODE
 * - 返回需要更新的信息（包含是否强制）
 */
public final class UpdateChecker {

    // TODO: 替换为你 CDN 上的固定地址（HTTPS）
    public static final String VERSION_URL = "https://cdn.xswy.tech/app/version.json"; // NEW

    // 新增：统一、安全地获取当前应用的 versionCode，避免 BuildConfig 包名不一致问题
    private static int getAppVersionCode(Context ctx) {
        try {
            // API 28+ 使用 longVersionCode
            android.content.pm.PackageManager pm = ctx.getPackageManager();
            android.content.pm.PackageInfo pi = pm.getPackageInfo(ctx.getPackageName(), 0);
            if (Build.VERSION.SDK_INT >= 28) {
                return (int) pi.getLongVersionCode();
            } else {
                return pi.versionCode;
            }
        } catch (Exception e) {
            // 新增：兜底返回一个很大的值，宁可不弹窗也不要误判为“必须更新”
            return Integer.MAX_VALUE;
        }
    }


    public interface Callback {
        /** info == null 表示无更新或拉取失败 */
        void onResult(UpdateInfo info);
    }

    public static final class UpdateInfo {
        public int latestCode;
        public int minSupported;
        public String latestName;
        public String notes;
        public String landingUrl; // 推荐：下载落地页（外部浏览器打开）
        public boolean force;     // 当前版本低于 minSupported → 强更
    }

    public static void check(Context ctx, Callback cb) {
        new Thread(() -> {
            try {
                HttpURLConnection conn = (HttpURLConnection) new URL(VERSION_URL).openConnection();
                conn.setConnectTimeout(5000);  // 超时
                conn.setReadTimeout(5000);     // 超时
                // 新增：禁用缓存，防止旧 JSON 误判
                conn.setUseCaches(false);
                conn.addRequestProperty("Cache-Control", "no-cache");
                conn.addRequestProperty("Pragma", "no-cache");

                int http = conn.getResponseCode();
                if (http != 200) {
                    // 网络失败策略——静默：返回 null
                    cb.onResult(null);
                    return;
                }

                String json = readAll(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));

                // MOD: 解析统一 JSON：先拿到 android 分区
                JSONObject root = new JSONObject(json);
                JSONObject a = root.getJSONObject("android");

                UpdateInfo info = new UpdateInfo();
                info.latestCode   = a.getInt("latestVersionCode");
                info.minSupported = a.getInt("minSupportedCode");
                info.latestName   = a.optString("latestVersionName", "");
                info.notes        = localizedNotes(ctx, a);
                info.landingUrl   = a.getString("landingUrl");

                int current = getAppVersionCode(ctx);  // 修改：改为通过 PackageManager 读取

                info.force = current < info.minSupported;

                if (current < info.latestCode) cb.onResult(info);
                else cb.onResult(null);

            } catch (Exception e) {
                // NEW: 任意异常 → 静默跳过
                cb.onResult(null);
            }
        }).start();
    }

    // 界面是日文、且 version.json 里填了 notes_ja 时，弹窗用日文说明；否则用 notes（没填 notes_ja 时与原来一样）
    private static String localizedNotes(Context ctx, JSONObject a) {
        String notesJa = a.optString("notes_ja", "");
        // 按系统语言列表挑界面语言，与资源匹配一致（如 [英语, 日语] 的用户看到的是日文界面）
        Locale ui = ctx.getResources().getConfiguration().getLocales()
                .getFirstMatch(new String[]{"zh", "ja"});
        boolean japanese = ui != null && "ja".equals(ui.getLanguage());
        return japanese && !notesJa.isEmpty() ? notesJa : a.optString("notes", "");
    }

    private static String readAll(Reader r) throws IOException {
        StringBuilder sb = new StringBuilder();
        char[] buf = new char[4096];
        int n;
        while ((n = r.read(buf)) != -1) sb.append(buf, 0, n);
        return sb.toString();
    }
}
