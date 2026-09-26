import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui' show ImageFilter;

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const PocketKittyApp());
}

class PocketKittyApp extends StatelessWidget {
  const PocketKittyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '口袋毛孩',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFC96442)),
      ),
      home: const HomePage(),
    );
  }
}

// ============================================================
// 原生桥接：iOS Vision Framework / Android ML Kit 主体分割
// ============================================================

class SegmentationService {
  static const MethodChannel _channel =
      MethodChannel('pet_segmentation/segment');

  /// 原生端构建标识。首页底部会显示；如果看不到它，说明装的是旧包
  static Future<String> nativeVersion() async {
    try {
      return await _channel.invokeMethod<String>('buildVersion') ?? 'unknown';
    } catch (_) {
      return 'old-apk';
    }
  }

  /// 输入原图路径，返回抠好背景（透明 PNG）的文件路径。
  /// [maskSource] 选择模型 7 路侧输出之一（0~6）：第 1 路是最终融合图，
  /// 其余是不同深度的边缘响应，真机上可切换对比选出边缘最好的一路。
  /// 失败时抛 [SegmentationException]，message 可直接展示给用户。
  static Future<String> removeBackground(String path, {int maskSource = 0}) async {
    try {
      final out = await _channel.invokeMethod<String>(
          'removeBackground', {'path': path, 'maskSource': maskSource});
      if (out == null || out.isEmpty) {
        throw const SegmentationException('原生端没有返回结果');
      }
      return out;
    } on PlatformException catch (e) {
      throw SegmentationException(e.message ?? '抠图失败（${e.code}）');
    } on MissingPluginException {
      throw const SegmentationException(
          '原生桥接未注册，请检查 AppDelegate / MainActivity 配置');
    }
  }
}

class SegmentationException implements Exception {
  const SegmentationException(this.message);
  final String message;

  @override
  String toString() => message;
}

// ============================================================
// 数据模型
// ============================================================

enum PetItemStatus { processing, done, failed }

class PetItem {
  PetItem({
    required this.originalPath,
    this.cutoutPath,
    this.status = PetItemStatus.processing,
    this.error,
  });

  final String originalPath;
  String? cutoutPath;
  PetItemStatus status;
  String? error;
}

// ============================================================
// 首页：选照片 → 自动抠图 → 展示
// ============================================================

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  final List<PetItem> _items = [];
  int _selected = 0;
  bool _picking = false;

  /// 边缘源（模型 7 路侧输出的 0~6），切换后自动重抠当前图
  int _maskSource = 0;

  /// 原生端构建标识：用于确认真机上装的是修复后的包
  late final Future<String> _nativeVersion = SegmentationService.nativeVersion();

  /// 用当前边缘源对当前图重新抠图（对比 d1~d7 时用）
  Future<void> _reprocessCurrent() async {
    final item = _items[_selected.clamp(0, _items.length - 1)];
    setState(() {
      item
        ..cutoutPath = null
        ..status = PetItemStatus.processing;
    });
    try {
      final out = await SegmentationService.removeBackground(
          item.originalPath,
          maskSource: _maskSource);
      if (!mounted) return;
      setState(() {
        item
          ..cutoutPath = out
          ..status = PetItemStatus.done;
      });
    } on SegmentationException catch (e) {
      if (!mounted) return;
      setState(() {
        item
          ..status = PetItemStatus.failed
          ..error = e.message;
      });
    }
  }

  Future<void> _pickAndProcess() async {
    if (_picking) return;
    setState(() => _picking = true);
    try {
      // 一次可多选，把三张「芝士」照片一起选中即可
      final pics = await _picker.pickMultiImage();
      if (pics == null || pics.isEmpty) return;

      final start = _items.length;
      setState(() {
        for (final p in pics) {
          _items.add(PetItem(originalPath: p.path));
        }
        _selected = start;
      });

      // 依次送到原生端抠图
      for (var i = start; i < _items.length; i++) {
        final item = _items[i];
        try {
          final out = await SegmentationService.removeBackground(
              item.originalPath,
              maskSource: _maskSource);
          if (!mounted) return;
          setState(() {
            item
              ..cutoutPath = out
              ..status = PetItemStatus.done;
          });
        } on SegmentationException catch (e) {
          if (!mounted) return;
          setState(() {
            item
              ..status = PetItemStatus.failed
              ..error = e.message;
          });
        }
      }
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFDF8F1),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('口袋毛孩 · 1.0 抠图测试'),
        actions: [
          if (_picking)
            const Padding(
              padding: EdgeInsets.only(right: 16),
              child: Center(
                child: SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
            )
          else
            IconButton(
              tooltip: '选择照片',
              onPressed: _pickAndProcess,
              icon: const Icon(Icons.add_photo_alternate_outlined),
            ),
        ],
      ),
      body: _items.isEmpty ? _buildEmpty() : _buildStage(),
    );
  }

  /// 首页底部的版本徽章：看到 v1.0.2-assetmgr = 新包；看到 old-apk = 旧包还在运行
  Widget _versionBadge() {
    return FutureBuilder<String>(
      future: _nativeVersion,
      builder: (context, snap) => Text(
        '原生端 ${snap.data ?? "…"}',
        style: TextStyle(fontSize: 10, color: Colors.brown.shade200),
      ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.pets, size: 72, color: Colors.brown.shade300),
            const SizedBox(height: 16),
            Text('先生成你的数字毛孩',
                style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(
              '选 2-3 张清晰照片（不同角度更佳），\n自动抠除背景，保留真实的它。',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.brown.shade400, height: 1.6),
            ),
            const SizedBox(height: 28),
            FilledButton.icon(
              onPressed: _pickAndProcess,
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('选择本地照片'),
            ),
            const SizedBox(height: 16),
            _versionBadge(),
          ],
        ),
      ),
    );
  }

  Widget _buildStage() {
    final item = _items[_selected.clamp(0, _items.length - 1)];
    return SafeArea(
      child: Column(
        children: [
          Expanded(child: PetStage(item: item)),
          _buildMaskSourceChips(),
          _buildThumbs(),
          const SizedBox(height: 8),
          Text(
            '点一下它试试 · 底部可切换角度',
            style: TextStyle(fontSize: 12, color: Colors.brown.shade300),
          ),
          const SizedBox(height: 2),
          _versionBadge(),
          const SizedBox(height: 4),
        ],
      ),
    );
  }

  /// 边缘源调试芯片：d1~d7 对应模型 7 路侧输出，点击即切换并重抠当前图
  Widget _buildMaskSourceChips() {
    return SizedBox(
      height: 44,
      child: Row(
        children: [
          const SizedBox(width: 16),
          Text('边缘源',
              style: TextStyle(fontSize: 11, color: Colors.brown.shade400)),
          const SizedBox(width: 8),
          Expanded(
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(vertical: 6),
              itemCount: 7,
              separatorBuilder: (_, __) => const SizedBox(width: 6),
              itemBuilder: (context, i) {
                final selected = i == _maskSource;
                return GestureDetector(
                  onTap: () {
                    if (_maskSource == i) return;
                    setState(() => _maskSource = i);
                    _reprocessCurrent();
                  },
                  child: Container(
                    width: 42,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: selected
                          ? Theme.of(context).colorScheme.primary
                          : Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: selected
                            ? Theme.of(context).colorScheme.primary
                            : Colors.brown.shade200,
                      ),
                    ),
                    child: Text(
                      'd${i + 1}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight:
                            selected ? FontWeight.w600 : FontWeight.w400,
                        color: selected ? Colors.white : Colors.brown.shade500,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          const SizedBox(width: 12),
        ],
      ),
    );
  }

  Widget _buildThumbs() {
    return SizedBox(
      height: 92,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: _items.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (context, i) {
          final item = _items[i];
          final selected = i == _selected;
          return GestureDetector(
            onTap: () => setState(() => _selected = i),
            child: Container(
              width: 76,
              height: 76,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: selected
                      ? Theme.of(context).colorScheme.primary
                      : Colors.brown.shade100,
                  width: selected ? 2 : 1,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.brown.withOpacity(0.08),
                    blurRadius: 8,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: item.status == PetItemStatus.processing
                  ? const Center(
                      child: SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    )
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(14),
                      child: Image.file(
                        File(item.cutoutPath ?? item.originalPath),
                        fit: BoxFit.cover,
                        width: double.infinity,
                        height: double.infinity,
                        errorBuilder: (_, __, ___) =>
                            const Icon(Icons.broken_image_outlined),
                      ),
                    ),
            ),
          );
        },
      ),
    );
  }
}

// ============================================================
// 舞台：呼吸动画 + 柔和落影 + 点击跳跃与叫声气泡
// ============================================================

class PetStage extends StatefulWidget {
  const PetStage({super.key, required this.item});

  final PetItem item;

  @override
  State<PetStage> createState() => _PetStageState();
}

class _PetStageState extends State<PetStage> with TickerProviderStateMixin {
  late final AnimationController _breathCtrl = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2800),
  )..repeat(reverse: true);
  late final CurvedAnimation _breath =
      CurvedAnimation(parent: _breathCtrl, curve: Curves.easeInOut);

  late final AnimationController _jumpCtrl = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 560),
  );

  final AudioPlayer _player = AudioPlayer();
  final Random _random = Random();
  Timer? _hideBubbleTimer;
  bool _showBubble = false;
  String _bubbleText = '喵～';

  static const List<String> _meows = ['喵～', '喵呜～', '喵嗷！', '喵？', '咕噜咕噜…'];

  @override
  void initState() {
    super.initState();
    _player.setVolume(0.9);
  }

  @override
  void dispose() {
    _breathCtrl.dispose();
    _jumpCtrl.dispose();
    _hideBubbleTimer?.cancel();
    _player.dispose();
    super.dispose();
  }

    Future<void> _poke() async {
    if (!_jumpCtrl.isAnimating) {
      _jumpCtrl.forward(from: 0);
    }
    setState(() {
      _bubbleText = _meows[_random.nextInt(_meows.length)];
      _showBubble = true;
    });
    _hideBubbleTimer?.cancel();
    _hideBubbleTimer = Timer(const Duration(milliseconds: 1500), () {
      if (mounted) setState(() => _showBubble = false);
    });

    // 叫声三级降级：assets 音频 → 原生系统提示音 → 纯气泡
    var played = false;
    try {
      await _player.setPlaybackRate(0.9 + _random.nextDouble() * 0.3);
      await _player.play(AssetSource('sounds/meow.mp3'));
      played = true;
    } catch (_) {
      // assets 未配置，尝试原生提示音
    }
    if (!played) {
      try {
        await const MethodChannel('pet_segmentation/segment')
            .invokeMethod('clickSound');
      } catch (_) {
        // 都失败就只显示气泡，不打扰用户
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final stageH = constraints.maxHeight;
        final stageW = constraints.maxWidth;
        final imgW = min<double>(stageW * 0.72, 340);
        final imgH = stageH * 0.60;

        return Stack(
          alignment: Alignment.center,
          children: [
            // 落影（垫在宠物后面，宠物跳起时影子留在地上）
            SizedBox(
              width: imgW,
              height: imgH,
              child: _buildGroundShadow(),
            ),
            // 宠物本体
            GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: _poke,
              child: AnimatedBuilder(
                animation: Listenable.merge([_breath, _jumpCtrl]),
                builder: (context, child) {
                  final t = _jumpCtrl.value;
                  final lift = -sin(pi * t) * 64.0;
                  final breath = 1.0 + 0.02 * _breath.value;
                  final stretchY = (1.0 + 0.10 * sin(pi * t)) * breath;
                  final squashX = (1.0 - 0.07 * sin(pi * t)) * breath;
                  return Transform(
                    alignment: Alignment.bottomCenter,
                    transform: Matrix4.identity()
                      ..translate(0.0, lift)
                      ..scale(squashX, stretchY),
                    child: child,
                  );
                },
                child: SizedBox(
                  width: imgW,
                  height: imgH,
                  child: _buildPetImage(),
                ),
              ),
            ),
            // 叫声气泡
            Positioned(
              top: stageH * 0.06,
              child: _buildBubble(),
            ),
            // 状态提示
            Positioned(
              left: 16,
              right: 16,
              bottom: 6,
              child: _buildStatusLine(),
            ),
          ],
        );
      },
    );
  }

  Widget _buildGroundShadow() {
    final item = widget.item;
    if (item.status == PetItemStatus.done && item.cutoutPath != null) {
      // 用同一张透明图做黑色剪影，模糊后向下偏移 = 贴合轮廓的柔和落影
      return Transform.translate(
        offset: const Offset(0, 14),
        child: Opacity(
          opacity: 0.25,
          child: ImageFiltered(
            imageFilter: ImageFilter.blur(sigmaX: 14, sigmaY: 14),
            child: ColorFiltered(
              colorFilter: const ColorFilter.mode(
                  Color(0xFF3B2A1A), BlendMode.srcATop),
              child: Image.file(
                File(item.cutoutPath!),
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) => const SizedBox.shrink(),
              ),
            ),
          ),
        ),
      );
    }
    // 原图 / 处理中：只留一团椭圆光斑
    return Align(
      alignment: Alignment.bottomCenter,
      child: Container(
        width: 180,
        height: 22,
        margin: const EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(999),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFF3B2A1A).withOpacity(0.18),
              blurRadius: 18,
              spreadRadius: 2,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPetImage() {
    final item = widget.item;
    if (item.status == PetItemStatus.done && item.cutoutPath != null) {
      return Image.file(
        File(item.cutoutPath!),
        fit: BoxFit.contain,
        errorBuilder: (_, __, ___) =>
            const Icon(Icons.broken_image_outlined, size: 48),
      );
    }
    // 处理中 / 失败：显示原图占位
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.brown.withOpacity(0.18),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.file(
              File(item.originalPath),
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) =>
                  const Icon(Icons.broken_image_outlined, size: 48),
            ),
            if (item.status == PetItemStatus.processing)
              Container(
                color: Colors.black.withOpacity(0.25),
                alignment: Alignment.center,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 10),
                    Text('正在抠图…',
                        style: TextStyle(color: Colors.white)),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildBubble() {
    return AnimatedOpacity(
      opacity: _showBubble ? 1 : 0,
      duration: const Duration(milliseconds: 220),
      child: AnimatedScale(
        scale: _showBubble ? 1 : 0.6,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOutBack,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: Colors.brown.withOpacity(0.12),
                blurRadius: 12,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Text(
            _bubbleText,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Color(0xFF5A4634),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildStatusLine() {
    final item = widget.item;
    String text;
    Color color;
    switch (item.status) {
      case PetItemStatus.processing:
        text = 'AI 正在把它从背景里抱出来…';
        color = Colors.brown;
        break;
      case PetItemStatus.done:
        text = '已生成 1:1 数字毛孩 · 点它有惊喜';
        color = Colors.green.shade700;
        break;
      case PetItemStatus.failed:
        text = '抠图失败：${item.error ?? '未知原因'}';
        color = Colors.red.shade600;
        break;
    }
    return Text(
      text,
      textAlign: TextAlign.center,
      style: TextStyle(fontSize: 12, color: color),
    );
  }
}
