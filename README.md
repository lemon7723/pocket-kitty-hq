# 口袋毛孩 · Flutter v1.1.0 高画质终极版

拍照 → 自动抠图 → 屏幕中央的一比一真实宠物。**本版为画质拉满版**：

- **全量版 U2-Net**（84MB，此前是 4.4MB 轻量版）：主体完整度、边缘判断显著提升
- **贴片推理**：大图自动切 2×2 带重叠窗口分别推理再融合，有效分辨率翻倍，胡须/毛发细节保留
- **百分位归一化 + S 曲线 + 盒模糊羽化**：背景灰雾归零，地砖阴影不再被放大，边缘半透明过渡自然
- **d1~d7 边缘源芯片**：模型 7 路侧输出在 App 内实时切换对比，选中即重抠，无需重新打包

## 目录

```
仓库根/
├── pubspec.yaml                                  # 版本 1.1.0+4
├── README.md
├── .github/workflows/build-apk.yml               # 云端打包（含模型自动下载）
├── lib/main.dart                                 # Flutter 端（含 d1~d7 切换芯片）
└── android/
    ├── build.gradle / settings.gradle / gradle.properties
    └── app/
        ├── build.gradle                          # TFLite 2.14.0 + exifinterface，无 Google 服务
        └── src/main/
            ├── AndroidManifest.xml               # 纯离线
            ├── assets/u2net.tflite               # ⚠️ 见下方说明：不用上传
            └── kotlin/com/pocketkitty/pocket_kitty/MainActivity.kt   # BUILD_TAG = v1.1.0-hq
```

## ⚠️ 模型文件（84MB）怎么处理——不用上传

**GitHub 网页上传单文件上限 25MB**，84MB 的模型直接拖网页会失败。本工程已经解决了这个问题：

- 云端打包流程会**自动检测并下载模型**（workflow 第一步「确保抠图模型存在」）
- 所以你**只上传代码文件，模型不用传**，打包时云端自动补齐
- 本地包里的 `u2net.tflite` 是给你本地构建用的，上传仓库时可以跳过它

如果你希望把模型也提交进仓库（比如方便别人 clone），用 Git LFS：

```bash
git lfs install
git lfs track "*.tflite"
git add .gitattributes android/app/src/main/assets/u2net.tflite
git commit -m "add model via LFS"
```

模型来源：https://huggingface.co/abhimanyu666/u2nettflite/resolve/main/u2net.tflite （Apache-2.0）

## 云端打包（不需要配任何本地环境）

1. GitHub 新建**空仓库**（不要勾选 README 初始化）
2. 把本地包里的全部文件拖进仓库上传（`u2net.tflite` 可跳过，`.github` 是隐藏文件夹要确认传上）
3. 仓库核对三条：根目录直接是 `pubspec.yaml`（不是套两层文件夹）；`lib/main.dart` 含 `nativeVersion`；`MainActivity.kt` 含 `v1.1.0-hq`
4. **Actions** → Build Android APK → **Run workflow** → 5-10 分钟 → 最新一次运行的 Artifacts 下载 APK
5. 构建日志里能看到「u2net.tflite（全量版 84MB）缺失，自动下载补齐」+ 模型文件列表确认

## 华为手机安装

1. APK 用微信/QQ/华为分享传到手机
2. 设置 → 安全 → 更多安全设置 → 安装外部来源应用 → 给传文件的 App 开允许
3. 被纯净模式拦截：设置 → 系统和更新 → 纯净模式 → 关闭后重装
4. **先卸载旧版再装新版**

## 验证新包生效

首页底部版本徽章应显示 **`原生端 v1.1.0-hq`**；任何报错自动带 `[v1.1.0-hq]` 前缀；
手机「设置 → 应用管理」里版本号应为 **1.1.0**。

## 画质调优指南

- **d1~d7 芯片**：d1 是模型最终融合图（默认最优起点）。如果毛发边缘发糊或断须，
  逐个点 d2~d7 对比——不同深度的侧输出边缘响应不同，挑你家猫最好看的那路
- **期望管理**：贴片推理已把有效分辨率翻倍，胡须是「以接近原生尺度进入模型」，
  但 320² 的模型本体决定了单根细胡须只能保留到「连续的半透明线条」级别，
  逐根清晰还原需要 1024 输入的 ISNet 级模型（ONNX 生态，TFLite 需自行转换，下一版可做）
- **切回轻量版**：把 `MainActivity.kt` 里 `MODEL_FILE` 改成 `"u2netp.tflite"`，
  workflow 会自动下载轻量模型（低配机卡顿时用）

## 已知边界

- 首次抠图需初始化 84MB 模型，比 v1.0.2 慢 2-3 秒，之后每次推理稳定
- 2×2 贴片在接缝处有余弦渐变融合，正常不可见；若在极均匀的背景上发现隐约直线的融合痕，
  把 `TILE_THRESHOLD` 从 900 调高到 1440 可关闭贴片（单窗口模式）
- iOS 分支不变：Vision 前景分割要求 iOS 17+
