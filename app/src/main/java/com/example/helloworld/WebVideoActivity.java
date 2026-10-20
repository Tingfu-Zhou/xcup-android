package com.example.helloworld;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.text.TextUtils;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.webkit.CookieManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 网页视频模式：
 *  - 内置 WebView 加载用户输入的视频网页
 *  - 嗅探页面里的视频流（.mp4 / .m3u8 / .m3u）
 *  - 嗅到后把 URL（含必要 headers）交给 VideoProcessActivity 走离线分析链路
 *  - 通过"蜜罐 WebView"模式吸收广告 window.open() 弹窗
 *  - 拦截跨站跳转，避免主窗口被甩到广告落地页
 *  - 拦截常见广告域名，清理页面
 */
public class WebVideoActivity extends AppCompatActivity {
    private static final String TAG = "WebVideoActivity";

    /** Intent extras 用来把嗅探到的 headers 传给 VideoProcessActivity */
    public static final String EXTRA_IS_WEB_VIDEO = "IS_WEB_VIDEO";
    public static final String EXTRA_HTTP_REFERER = "HTTP_REFERER";
    public static final String EXTRA_HTTP_UA = "HTTP_USER_AGENT";
    public static final String EXTRA_HTTP_COOKIE = "HTTP_COOKIE";
    /** WebView 里 &lt;video&gt;.currentTime 的当前播放位置（ms），分析页开局 seek 到这里 */
    public static final String EXTRA_START_POSITION_MS = "START_POSITION_MS";

    /** 「抓到的可能是广告」提示卡被用户关掉后记一笔，以后不再显示 */
    private static final String PREFS = "web_video";
    private static final String KEY_AD_TIP_DISMISSED = "ad_tip_dismissed";

    /** 常见广告/分析域名黑名单。命中即在 shouldInterceptRequest 里返回空响应。 */
    private static final Set<String> AD_HOSTS = new HashSet<>(Arrays.asList(
            "ero-labs.top",
            "trafficstars.com",
            "tsyndicate.com",
            "juicyads.com",
            "exoclick.com",
            "exosrv.com",
            "adsterra.net",
            "adsterra.com",
            "popads.net",
            "revcontent.com",
            "doubleclick.net",
            "googlesyndication.com"
    ));

    /** 桌面 Chrome UA — 多数视频站会按 UA 区分 */
    private static final String DESKTOP_UA =
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    + "Chrome/120.0.0.0 Safari/537.36";

    private static final long JS_POLL_INTERVAL_MS = 2000;

    /**
     * 注入页面的嗅探 JS。每次 onPageStarted 调一次。
     *
     * 做三件事：
     *  1) hook HTMLMediaElement.prototype.src 的 setter，src 一变就上报；
     *  2) hook HTMLSourceElement.prototype.src，同理；
     *  3) 每 1.5 秒扫一遍 DOM，挑出"面积最大 + 正在播 + 非静音"那个 <video>，
     *     报它的 currentSrc。
     *
     * 每个 video 元素分配一个 __sniffId，让 native 端能识别"同一个元素改了 src"
     * （典型场景：Pornhub 主播放器里广告→真视频）。
     */
    private static final String SNIFF_JS =
            "(function(){\n" +
            "  if (window.__sniffInstalled) return;\n" +
            "  window.__sniffInstalled = true;\n" +
            "  function nextId(){ window.__sniffN=(window.__sniffN||0)+1; return '__el_'+window.__sniffN; }\n" +
            "  function elInfo(el){\n" +
            "    if(!el) return {width:0,height:0,paused:true,muted:true,currentTime:0,elementId:''};\n" +
            "    if(!el.__sniffId) el.__sniffId = nextId();\n" +
            "    var r = el.getBoundingClientRect();\n" +
            "    return {width:r.width||0, height:r.height||0,\n" +
            "            paused: !!el.paused, muted: !!el.muted,\n" +
            "            currentTime: el.currentTime||0, elementId: el.__sniffId};\n" +
            "  }\n" +
            "  function report(url, source, el){\n" +
            "    try{\n" +
            "      if(!url || typeof url!=='string') return;\n" +
            "      if(url.indexOf('http')!==0) return;\n" +
            "      var info = elInfo(el);\n" +
            "      info.url = url; info.source = source;\n" +
            "      window.AndroidVideoSniff.onVideoSrc(JSON.stringify(info));\n" +
            "    }catch(e){}\n" +
            "  }\n" +
            "  function hookSrc(proto, label){\n" +
            "    try{\n" +
            "      var d = Object.getOwnPropertyDescriptor(proto, 'src');\n" +
            "      if(!d || !d.set || !d.get) return;\n" +
            "      Object.defineProperty(proto, 'src', {\n" +
            "        configurable: true,\n" +
            "        get: function(){ return d.get.call(this); },\n" +
            "        set: function(v){\n" +
            "          try{\n" +
            "            var el = this;\n" +
            "            if(label==='source-src'){ var p=this.parentElement; if(p && p.tagName==='VIDEO') el=p; }\n" +
            "            report(v, label, el);\n" +
            "          }catch(e){}\n" +
            "          return d.set.call(this, v);\n" +
            "        }\n" +
            "      });\n" +
            "    }catch(e){}\n" +
            "  }\n" +
            "  hookSrc(HTMLMediaElement.prototype, 'media-src');\n" +
            "  if(window.HTMLSourceElement) hookSrc(HTMLSourceElement.prototype, 'source-src');\n" +
            "  function scan(){\n" +
            "    try{\n" +
            "      var vs = document.querySelectorAll('video');\n" +
            "      var best=null, bestScore=0;\n" +
            "      for(var i=0;i<vs.length;i++){\n" +
            "        var v = vs[i];\n" +
            "        var r = v.getBoundingClientRect();\n" +
            "        if(r.width<200 || r.height<100) continue;\n" +
            "        if(r.bottom<0 || r.top>window.innerHeight) continue;\n" +
            "        var area = r.width*r.height;\n" +
            "        var w = 1;\n" +
            "        if(!v.paused && v.currentTime>0) w *= 100;\n" +
            "        if(!v.muted) w *= 10;\n" +
            "        var s = area*w;\n" +
            "        if(s>bestScore){ bestScore=s; best=v; }\n" +
            "      }\n" +
            "      if(best){\n" +
            "        var src = best.src || best.currentSrc || '';\n" +
            "        if(src && src.indexOf('http')===0) report(src, 'dom-scan', best);\n" +
            "      }\n" +
            "    }catch(e){}\n" +
            "  }\n" +
            "  scan();\n" +
            "  setInterval(scan, 1500);\n" +
            "})();";

    private WebView webView;
    private TextInputEditText etUrl;
    private MaterialButton btnOpen;
    private MaterialButton btnAnalyze;
    private TextView tvStatus;

    private final AtomicReference<String> detectedVideoUrl = new AtomicReference<>(null);
    private final AtomicReference<String> detectedReferer = new AtomicReference<>(null);
    private final AtomicReference<String> detectedUa = new AtomicReference<>(null);

    /** 嗅探打分 + 来源元素跟踪。所有跨线程读写都走 sniffLock。 */
    private final Object sniffLock = new Object();
    private double currentBestScore = 0;
    private String currentBestElementId = "";

    private Handler jsPollHandler;
    private Runnable jsPollRunnable;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_web_video);

        MaterialToolbar toolbar = findViewById(R.id.topAppBar);
        toolbar.setNavigationOnClickListener(v -> finish());

        webView = findViewById(R.id.webView);
        etUrl = findViewById(R.id.etUrl);
        btnOpen = findViewById(R.id.btnOpen);
        btnAnalyze = findViewById(R.id.btnAnalyze);
        tvStatus = findViewById(R.id.tvStatus);

        setupAdTip();
        configureWebView();

        btnOpen.setOnClickListener(v -> openCurrentUrl());

        etUrl.setOnEditorActionListener((tv, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_GO
                    || (event != null && event.getKeyCode() == KeyEvent.KEYCODE_ENTER)) {
                openCurrentUrl();
                return true;
            }
            return false;
        });

        btnAnalyze.setOnClickListener(v -> launchAnalysis());

        jsPollHandler = new Handler(Looper.getMainLooper());
        jsPollRunnable = new Runnable() {
            @Override
            public void run() {
                pollDomForVideoSrc();
                jsPollHandler.postDelayed(this, JS_POLL_INTERVAL_MS);
            }
        };
        jsPollHandler.postDelayed(jsPollRunnable, JS_POLL_INTERVAL_MS);
    }

    /**
     * 「抓到的可能是广告」提示卡：部分网站先插播广告，嗅探到的可能是广告片段，提示等正片再分析，
     * 或改用在线分析（录屏）。用户点关闭后不再显示（记在 SharedPreferences 里）。
     */
    private void setupAdTip() {
        View cardAdTip = findViewById(R.id.cardAdTip);
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        if (prefs.getBoolean(KEY_AD_TIP_DISMISSED, false)) {
            cardAdTip.setVisibility(View.GONE);
            return;
        }
        findViewById(R.id.btnCloseAdTip).setOnClickListener(v -> {
            cardAdTip.setVisibility(View.GONE);
            prefs.edit().putBoolean(KEY_AD_TIP_DISMISSED, true).apply();
        });
    }

    private void openCurrentUrl() {
        CharSequence raw = etUrl.getText();
        if (raw == null) return;
        String url = raw.toString().trim();
        if (TextUtils.isEmpty(url)) {
            Toast.makeText(this, R.string.web_video_status_idle, Toast.LENGTH_SHORT).show();
            return;
        }
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            url = "https://" + url;
        }
        resetDetection();
        tvStatus.setText(R.string.web_video_status_loading);
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null) {
            imm.hideSoftInputFromWindow(etUrl.getWindowToken(), 0);
        }
        webView.loadUrl(url);
    }

    private void resetDetection() {
        detectedVideoUrl.set(null);
        detectedReferer.set(null);
        detectedUa.set(null);
        synchronized (sniffLock) {
            currentBestScore = 0;
            currentBestElementId = "";
        }
        btnAnalyze.setVisibility(View.GONE);
    }

    private void launchAnalysis() {
        if (detectedVideoUrl.get() == null) return;

        // 先异步读 <video>.currentTime，作为分析页开局 seek 位置；不论成功失败都继续往下
        try {
            webView.evaluateJavascript(
                    "(function(){var vs=document.querySelectorAll('video');"
                            + "for(var i=0;i<vs.length;i++){var v=vs[i];"
                            + "if(v.currentTime>0)return v.currentTime;}return 0;})()",
                    value -> {
                        long startMs = parseCurrentTimeToMs(value);
                        doLaunchAnalysis(startMs);
                    });
        } catch (Exception e) {
            Log.w(TAG, "evaluateJavascript currentTime 失败，从 0 开始", e);
            doLaunchAnalysis(0L);
        }
    }

    /** 解析 evaluateJavascript 回来的字符串到毫秒。失败一律返回 0。 */
    private static long parseCurrentTimeToMs(String value) {
        if (value == null || value.isEmpty() || "null".equalsIgnoreCase(value)) return 0L;
        // value 可能形如 "12.345" 或带引号 "\"12.345\""
        String trimmed = value;
        if (trimmed.length() >= 2 && trimmed.startsWith("\"") && trimmed.endsWith("\"")) {
            trimmed = trimmed.substring(1, trimmed.length() - 1);
        }
        try {
            double secs = Double.parseDouble(trimmed);
            if (Double.isNaN(secs) || secs < 0) return 0L;
            return (long) (secs * 1000.0);
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private void doLaunchAnalysis(long startPositionMs) {
        String videoUrl = detectedVideoUrl.get();
        if (videoUrl == null) return;

        // 把网页视频暂停掉，避免 WebView 和 VideoProcessActivity 同时拉流
        webView.evaluateJavascript(
                "document.querySelectorAll('video,audio').forEach(function(v){try{v.pause();}catch(e){}});",
                null);

        Intent intent = new Intent(this, VideoProcessActivity.class);
        intent.setData(Uri.parse(videoUrl));
        intent.putExtra(EXTRA_IS_WEB_VIDEO, true);
        intent.putExtra(EXTRA_START_POSITION_MS, startPositionMs);

        String referer = detectedReferer.get();
        if (referer != null) intent.putExtra(EXTRA_HTTP_REFERER, referer);

        String ua = detectedUa.get();
        if (ua == null) ua = DESKTOP_UA;
        intent.putExtra(EXTRA_HTTP_UA, ua);

        try {
            // surrit.com（missav CDN）在 Cloudflare 后面，靠 __cf_bm 这类 cookie 区分人/机。
            // WebView 是真浏览器、已经过了 Cloudflare 校验并拿到了 cookie；把它带给 ExoPlayer，
            // 让重放请求尽量"接得上"那个已被放行的会话。flush 一下确保读到最新写入的 cookie。
            CookieManager.getInstance().flush();
            String cookie = CookieManager.getInstance().getCookie(videoUrl);
            if (!TextUtils.isEmpty(cookie)) {
                intent.putExtra(EXTRA_HTTP_COOKIE, cookie);
            }
            Log.i(TAG, "网页视频 Cookie for " + Uri.parse(videoUrl).getHost() + ": "
                    + (TextUtils.isEmpty(cookie) ? "<空>（没抓到 __cf_bm，cookie 重放这条路走不通）" : cookie));
        } catch (Exception e) {
            Log.w(TAG, "读取 Cookie 失败", e);
        }

        Log.i(TAG, "启动 VideoProcessActivity, url=" + videoUrl
                + ", referer=" + detectedReferer.get()
                + ", ua=" + ua
                + ", startMs=" + startPositionMs);
        String toast = startPositionMs > 0
                ? getString(R.string.web_video_opening_from, startPositionMs / 1000)
                : getString(R.string.web_video_opening);
        Toast.makeText(this, toast, Toast.LENGTH_SHORT).show();
        startActivity(intent);
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void configureWebView() {
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        s.setUserAgentString(DESKTOP_UA);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        s.setBuiltInZoomControls(true);
        s.setDisplayZoomControls(false);
        // 多窗口蜜罐
        s.setSupportMultipleWindows(true);
        s.setJavaScriptCanOpenWindowsAutomatically(true);

        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true);

        // 把 native 桥暴露给页面 JS，对应 SNIFF_JS 里的 AndroidVideoSniff.onVideoSrc(...)
        webView.addJavascriptInterface(new VideoSniffInterface(), "AndroidVideoSniff");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                // 每次进入新页面都重置嗅探状态
                resetDetection();
                tvStatus.setText(R.string.web_video_status_loading);
                // 尽可能早地把 hook 注入新文档。__sniffInstalled 守卫保证重复注入安全
                view.evaluateJavascript(SNIFF_JS, null);
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri reqUri = request.getUrl();
                if (reqUri == null) return false;
                String reqHost = reqUri.getHost();
                String pageUrl = view.getUrl();
                if (reqHost == null || pageUrl == null) return false;
                Uri pageUri = Uri.parse(pageUrl);
                String pageHost = pageUri.getHost();
                if (pageHost == null) return false;

                String pageRoot = extractRootDomain(pageHost);
                String reqRoot = extractRootDomain(reqHost);
                if (!reqRoot.equalsIgnoreCase(pageRoot)) {
                    Log.d(TAG, "拦截跨站跳转: " + reqUri + " (page=" + pageHost + ")");
                    return true;
                }
                return false;
            }

            @Nullable
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                Uri reqUri = request.getUrl();
                if (reqUri == null) return null;
                String reqHost = reqUri.getHost();
                String reqUrl = reqUri.toString();

                // 1) 广告域名直接屏蔽
                if (reqHost != null && isAdHost(reqHost)) {
                    Log.d(TAG, "拦截广告域名: " + reqUri);
                    return blankResponse();
                }

                // 2) 视频流嗅探 —— 网络层不知道是哪个 <video> 元素发出的请求，
                //    打最低分（1）作为兜底；只要页面里有 JS 报上更高分的来源，这条就会被压住
                if (looksLikeVideoUrl(reqUrl)) {
                    String referer = null;
                    String ua = null;
                    try {
                        if (request.getRequestHeaders() != null) {
                            referer = request.getRequestHeaders().get("Referer");
                            ua = request.getRequestHeaders().get("User-Agent");
                        }
                    } catch (Exception ignored) { }

                    handleSniffedVideo(
                            reqUrl,
                            /*score*/ 1.0,
                            "network",
                            /*elementId*/ "",
                            /*networkReferer*/ referer,
                            /*networkUa*/ ua);
                }

                // 返回 null = 不拦截，让请求正常进行
                return null;
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onCreateWindow(WebView view, boolean isDialog,
                                          boolean isUserGesture, Message resultMsg) {
                // "蜜罐 WebView"：把弹窗导向一个临时 WebView，并立刻吃掉里面任何跳转
                final WebView popup = new WebView(view.getContext());
                WebSettings ps = popup.getSettings();
                ps.setJavaScriptEnabled(true);
                popup.setWebViewClient(new WebViewClient() {
                    @Override
                    public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest req) {
                        Log.d(TAG, "拦截弹窗跳转: " + req.getUrl());
                        // 推迟到下一拍销毁，避免在 WebView 自己的回调里 destroy
                        v.post(popup::destroy);
                        return true;
                    }
                });
                WebView.WebViewTransport transport = (WebView.WebViewTransport) resultMsg.obj;
                transport.setWebView(popup);
                resultMsg.sendToTarget();
                return true;
            }

            @Override
            public void onCloseWindow(WebView window) {
                window.post(window::destroy);
            }
        });
    }

    /** 老的兜底 JS 轮询，保留是因为有些站点 setInterval 注入失败时还能找回主视频 src。 */
    private void pollDomForVideoSrc() {
        try {
            webView.evaluateJavascript(
                    "(function(){var vs=document.querySelectorAll('video');for(var i=0;i<vs.length;i++){"
                            + "var v=vs[i];if(v.src&&v.src.length>0)return v.src;"
                            + "if(v.currentSrc&&v.currentSrc.length>0)return v.currentSrc;}return '';})()",
                    value -> {
                        if (value == null) return;
                        String src = value;
                        if (src.length() >= 2 && src.startsWith("\"") && src.endsWith("\"")) {
                            src = src.substring(1, src.length() - 1);
                        }
                        src = src.replace("\\/", "/");
                        if (src.startsWith("http")) {
                            // 走统一打分入口；这条路径没有元素几何，用最低分作为兜底
                            handleSniffedVideo(src, /*score*/ 1.0, "fallback-poll",
                                    /*elementId*/ "", /*networkReferer*/ null, /*networkUa*/ null);
                        }
                    });
        } catch (Exception e) {
            Log.w(TAG, "JS 嗅探异常", e);
        }
    }

    /**
     * 嗅到一个视频 URL 时的统一入口。打分 + 决定要不要覆盖当前已锁定的 URL。
     *
     * 来源（source）有：
     *  - media-src     : 页面里某个 &lt;video&gt;.src = "..." 被赋值
     *  - source-src    : 页面里某个 &lt;source&gt;.src = "..." 被赋值
     *  - dom-scan      : in-page setInterval 挑出的"最大正在播放"那个 &lt;video&gt;
     *  - fallback-poll : native 端 evaluateJavascript 兜底
     *  - network       : shouldInterceptRequest 抓到（不知道是哪个元素）
     *
     * 更新规则：
     *  - 同一 URL → 跳过
     *  - 同一 video 元素（同一个 __sniffId）改了 src → 总是更新（典型：Pornhub 广告→真视频）
     *  - 新 URL 分数 &gt;= 当前最高 → 更新
     *  - 其它情况 → 跳过
     */
    private void handleSniffedVideo(String url, double score, String source,
                                    String elementId,
                                    String networkReferer, String networkUa) {
        if (!looksLikeVideoUrl(url)) return;
        if (looksLikePreviewUrl(url)) {
            Log.d(TAG, "跳过预览/trickplay URL (" + source + "): " + url);
            return;
        }

        boolean shouldUpdate;
        double prevScore;
        synchronized (sniffLock) {
            String prev = detectedVideoUrl.get();
            if (url.equals(prev)) return;   // 同一 URL 不重复

            prevScore = currentBestScore;
            boolean sameElement = elementId != null && !elementId.isEmpty()
                    && elementId.equals(currentBestElementId);

            if (prev == null) {
                shouldUpdate = true;
            } else if (sameElement) {
                // 同一元素 src 变了（Pornhub 主播放器从广告切到真视频），绕过分数比较
                shouldUpdate = true;
            } else if (score > currentBestScore) {
                // 严格大于。相同分数（如多条 network=1）不允许互相覆盖，
                // 真视频先到先得；这样 missav 主视频 playlist 不会被推荐位的 playlist 顶下去
                shouldUpdate = true;
            } else {
                shouldUpdate = false;
            }

            if (shouldUpdate) {
                currentBestScore = score;
                currentBestElementId = elementId == null ? "" : elementId;
                detectedVideoUrl.set(url);

                if (networkReferer != null && !networkReferer.isEmpty()) {
                    detectedReferer.set(networkReferer);
                } else if (detectedReferer.get() == null) {
                    String pageUrl = webView.getUrl();
                    if (pageUrl != null) detectedReferer.set(pageUrl);
                }
                if (networkUa != null && !networkUa.isEmpty()) {
                    detectedUa.set(networkUa);
                } else if (detectedUa.get() == null) {
                    detectedUa.set(DESKTOP_UA);
                }
            }
        }

        if (shouldUpdate) {
            Log.i(TAG, String.format("更新视频流 (score=%d, %s): %s",
                    (long) score, source, url));
            final String finalUrl = url;
            final String finalSource = source;
            final long finalScore = (long) score;
            runOnUiThread(() -> {
                tvStatus.setText(getString(R.string.web_video_status_found)
                        + " [" + finalSource + ", score=" + finalScore + "]\n" + finalUrl);
                btnAnalyze.setVisibility(View.VISIBLE);
            });
        } else {
            Log.d(TAG, String.format("跳过低分 URL (score=%d < %d, %s): %s",
                    (long) score, (long) prevScore, source, url));
        }
    }

    /** WebView 暴露给页面 JS 的桥，对应 SNIFF_JS 里的 AndroidVideoSniff.onVideoSrc(...) */
    private class VideoSniffInterface {
        @JavascriptInterface
        public void onVideoSrc(String json) {
            if (json == null) return;
            try {
                JSONObject o = new JSONObject(json);
                String url = o.optString("url", "");
                String source = o.optString("source", "unknown");
                double width = o.optDouble("width", 0);
                double height = o.optDouble("height", 0);
                boolean paused = o.optBoolean("paused", true);
                boolean muted = o.optBoolean("muted", true);
                double currentTime = o.optDouble("currentTime", 0);
                String elementId = o.optString("elementId", "");

                double area = width * height;
                if (area < 200 * 100) {
                    // 元素太小，必是侧栏 / hover 缩略图，直接丢
                    Log.d(TAG, "跳过太小元素 (" + (long) width + "x" + (long) height
                            + ", " + source + "): " + url);
                    return;
                }

                double weight = 1.0;
                if (!paused && currentTime > 0) weight *= 100.0;   // 正在播
                if (!muted) weight *= 10.0;                        // 有声音（推荐位多半静音）
                double score = area * weight;

                handleSniffedVideo(url, score, source, elementId, null, null);
            } catch (Exception e) {
                Log.w(TAG, "VideoSniffInterface.onVideoSrc 解析失败: " + json, e);
            }
        }
    }

    /**
     * "预览/trickplay/缩略图条" URL 黑名单。这种 URL 也是 .mp4 / .m3u8 后缀，
     * 但播放器拉到只有几秒、分辨率极小，不是真视频。
     *
     * 触发条件命中任一即跳过：
     *   1) 子域名暗示：pix- / thumb- / sprite- / preview- / trickplay- 开头
     *   2) 路径段：/preview/ /thumb/ /sprite/ /storyboard/ /poster/ /trickplay/ /vts:
     *   3) Pornhub 风格的 vts:数字 标记
     *   4) Pornhub 风格的 rs:fit:W:H 且 W*H 太小（< 240000 ≈ 600x400 以下）
     */
    private static boolean looksLikePreviewUrl(String url) {
        if (url == null) return false;
        String lower = url.toLowerCase(Locale.US);

        // 路径标记
        if (lower.contains("/vts:") || lower.contains("/vts/")
                || lower.contains("/preview/") || lower.contains("/thumb/")
                || lower.contains("/thumbnail/") || lower.contains("/sprite/")
                || lower.contains("/storyboard/") || lower.contains("/poster/")
                || lower.contains("/trickplay/") || lower.contains("/seek-thumb/")
                || lower.contains("?type=preview") || lower.contains("&type=preview")) {
            return true;
        }

        // Pornhub 风格 vts:NNN —— vts: 后面跟数字
        int vtsIdx = lower.indexOf("vts:");
        if (vtsIdx >= 0 && vtsIdx + 4 < lower.length()
                && Character.isDigit(lower.charAt(vtsIdx + 4))) {
            return true;
        }

        // 子域名暗示
        try {
            Uri uri = Uri.parse(url);
            String host = uri.getHost();
            if (host != null) {
                String h = host.toLowerCase(Locale.US);
                if (h.startsWith("pix-") || h.startsWith("thumb-")
                        || h.startsWith("sprite-") || h.startsWith("preview-")
                        || h.startsWith("trickplay-")) {
                    return true;
                }
            }
        } catch (Exception ignored) { }

        // rs:fit:W:H —— 解析 W*H，太小判为预览
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("rs:fit:(\\d+):(\\d+)")
                .matcher(lower);
        if (m.find()) {
            try {
                long w = Long.parseLong(m.group(1));
                long h = Long.parseLong(m.group(2));
                if (w * h > 0 && w * h < 240_000L) return true;
            } catch (NumberFormatException ignored) { }
        }

        return false;
    }

    /**
     * fragmented MP4 / DASH-CMAF 分片识别。HLS 现在也常用 fMP4 替代 .ts，
     * 文件名是 .mp4 但只是个 1-2 秒的分片，单独喂给 ExoPlayer 会报
     * UnrecognizedInputFormatException。
     *
     * 典型样本：
     *   xxx_240p_h264_init_<token>.mp4           ← 初始化段
     *   xxx_240p_h264_589_<token>_1780755790.mp4 ← 媒体分片（带 codec 标记 + 序号）
     *   .../seg-15.mp4   .../segment-3.mp4       ← 通用编号分片
     */
    private static boolean looksLikeHlsFragment(String url) {
        if (url == null) return false;
        String lower = url.toLowerCase(Locale.US);
        int qIdx = lower.indexOf('?');
        String path = qIdx > 0 ? lower.substring(0, qIdx) : lower;
        if (!path.endsWith(".mp4")) return false;

        int slashIdx = path.lastIndexOf('/');
        String filename = slashIdx >= 0 ? path.substring(slashIdx + 1) : path;

        // 1) init segment：文件名带 "init"
        if (filename.contains("init.mp4")
                || filename.contains("_init_") || filename.contains("-init-")
                || filename.contains("_init.") || filename.contains("-init.")
                || filename.startsWith("init.") || filename.startsWith("init-")
                || filename.startsWith("init_")
                || filename.contains("initialization")) {
            return true;
        }

        // 2) 带 codec 标记 + 数字序号的分片
        //    e.g. xxx_h264_589_<token>.mp4 / xxx_h265_10.mp4 / xxx_av1_42.mp4
        if (filename.matches(".*_(h264|h265|hevc|avc|vp9|av01?|aac)_\\d+[._\\-].*\\.mp4")) {
            return true;
        }

        // 3) seg-N / segment-N / chunk-N / frag-N / fragment-N 命名
        if (filename.matches(".*(^|[_\\-/])(seg|segment|chunk|frag|fragment)[_\\-]?\\d+.*\\.mp4")) {
            return true;
        }

        return false;
    }

    private static boolean looksLikeVideoUrl(String url) {
        if (url == null) return false;
        String lower = url.toLowerCase(Locale.US);
        int qIdx = lower.indexOf('?');
        String path = qIdx > 0 ? lower.substring(0, qIdx) : lower;

        // HLS / DASH 分片单独都不能播，明确排除掉，避免被下面的容器扩展名误判
        if (path.endsWith(".ts") || path.endsWith(".m4s")
                || path.endsWith(".m4a") || path.endsWith(".aac")
                || path.endsWith(".vtt")) {
            return false;
        }

        // fragmented MP4 分片（后缀 .mp4 但其实是 HLS/DASH chunk，不能单独播）
        if (looksLikeHlsFragment(url)) return false;

        // 只认完整容器/playlist 的后缀。注意不要再用 contains(".mp4/") 之类的子串，
        // 因为 Pornhub 这种站把 ".mp4/" 当作目录段在 HLS 分片 URL 里复用，
        // 会把 ".../X.mp4/seg-N-v1-a1.ts" 误判为视频 URL。
        return path.endsWith(".mp4")
                || path.endsWith(".m3u8")
                || path.endsWith(".m3u")
                || path.endsWith(".webm")
                || path.endsWith(".mpd");   // DASH playlist
    }

    private static boolean isAdHost(String host) {
        String lower = host.toLowerCase(Locale.US);
        for (String ad : AD_HOSTS) {
            if (lower.equals(ad) || lower.endsWith("." + ad)) {
                return true;
            }
        }
        return false;
    }

    private static String extractRootDomain(String host) {
        String[] parts = host.split("\\.");
        if (parts.length >= 2) {
            return parts[parts.length - 2] + "." + parts[parts.length - 1];
        }
        return host;
    }

    private static WebResourceResponse blankResponse() {
        return new WebResourceResponse("text/plain", "utf-8",
                new ByteArrayInputStream(new byte[0]));
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (webView != null) {
            webView.onPause();
            webView.pauseTimers();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (webView != null) {
            webView.resumeTimers();
            webView.onResume();
        }
    }

    @Override
    protected void onDestroy() {
        if (jsPollHandler != null && jsPollRunnable != null) {
            jsPollHandler.removeCallbacks(jsPollRunnable);
        }
        if (webView != null) {
            webView.loadUrl("about:blank");
            webView.clearHistory();
            webView.removeAllViews();
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }
}
