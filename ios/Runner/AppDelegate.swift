import UIKit
import Flutter
import Vision
import CoreVideo

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)

    let controller = window?.rootViewController as! FlutterViewController
    let channel = FlutterMethodChannel(
      name: "pet_segmentation/segment",
      binaryMessenger: controller.binaryMessenger
    )

    channel.setMethodCallHandler { call, result in
      switch call.method {
      case "removeBackground":
        guard let args = call.arguments as? [String: Any],
              let path = args["path"] as? String else {
          result(FlutterError(code: "BAD_ARGS", message: "缺少图片路径", details: nil))
          return
        }
        DispatchQueue.global(qos: .userInitiated).async {
          PocketSegmenter.removeBackground(path: path) { outPath, error in
            DispatchQueue.main.async {
              if let outPath = outPath {
                result(outPath)
              } else {
                result(FlutterError(code: "SEGMENT_FAIL", message: error ?? "抠图失败", details: nil))
              }
            }
          }
        }
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}

// ============================================================
// iOS 自动抠图：Vision Framework 前景实例分割（iOS 17+）
// 输入：任意带背景的宠物照片路径
// 输出：透明底 PNG 文件路径
// ============================================================

enum PocketSegmenter {

  static func removeBackground(path: String, completion: @escaping (String?, String?) -> Void) {
    guard #available(iOS 17.0, *) else {
      completion(nil, "自动抠图需要 iOS 17 或更高版本（使用了 Vision 前景分割 API）")
      return
    }

    guard let source = UIImage(contentsOfFile: path) else {
      completion(nil, "无法读取图片文件")
      return
    }

    // 统一方向（EXIF）并控制尺寸，避免大图占满内存
    let normalized = normalizedUp(source)
    let scaled = downscale(normalized, maxDim: 2048)
    guard let cgImage = scaled.cgImage else {
      completion(nil, "无法读取图片数据")
      return
    }

    let request = VNGenerateForegroundInstanceMaskRequest()
    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])

    do {
      try handler.perform([request])

      guard let observation = request.results?.first else {
        completion(nil, "没有识别到明确的主体，试试更清晰、宠物占比更大的照片")
        return
      }

      let maskBuffer = try observation.generateScaledMaskForImage(
        forInstances: observation.allInstances,
        from: handler
      )

      guard let pngData = composeTransparentPNG(cgImage: cgImage, mask: maskBuffer) else {
        completion(nil, "生成透明底图片失败")
        return
      }

      let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("cutout_\(Int(Date().timeIntervalSince1970 * 1000)).png")
      try pngData.write(to: url)
      completion(url.path, nil)
    } catch {
      completion(nil, "抠图失败：\(error.localizedDescription)")
    }
  }

  /// 把原图 CGImage 与 Vision 生成的单通道遮罩合成为「预乘 Alpha」的透明底 PNG。
  /// 直接改 alpha 会产生毛边光晕，所以 RGB 需要按 alpha 预乘。
  @available(iOS 17.0, *)
  private static func composeTransparentPNG(cgImage: CGImage, mask: CVPixelBuffer) -> Data? {
    let width = CVPixelBufferGetWidth(mask)
    let height = CVPixelBufferGetHeight(mask)

    guard let ctx = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    ctx.interpolationQuality = .high
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let base = ctx.data else { return nil }
    let pixels = base.bindMemory(to: UInt8.self, capacity: width * height * 4)

    CVPixelBufferLockBaseAddress(mask, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
    guard let maskBase = CVPixelBufferGetBaseAddress(mask) else { return nil }

    let format = CVPixelBufferGetPixelFormatType(mask)
    let maskRowBytes = CVPixelBufferGetBytesPerRow(mask)

    for y in 0..<height {
      let maskRowStart = y * maskRowBytes
      let pixelRowStart = y * width * 4
      for x in 0..<width {
        var alpha: UInt8
        switch format {
        case kCVPixelFormatType_OneComponent8:
          alpha = maskBase.load(from: maskRowStart + x, as: UInt8.self)
        case kCVPixelFormatType_32Float:
          let v = maskBase.load(from: maskRowStart + x * 4, as: Float.self)
          alpha = UInt8(max(0, min(255, round(v * 255.0))))
        default:
          alpha = 255
        }

        let idx = pixelRowStart + x * 4
        if alpha < 255 {
          let k = Float(alpha) / 255.0
          pixels[idx]     = UInt8(Float(pixels[idx]) * k)
          pixels[idx + 1] = UInt8(Float(pixels[idx + 1]) * k)
          pixels[idx + 2] = UInt8(Float(pixels[idx + 2]) * k)
        }
        pixels[idx + 3] = alpha
      }
    }

    guard let outCG = ctx.makeImage() else { return nil }
    return UIImage(cgImage: outCG).pngData()
  }

  /// 按 EXIF 方向把图片转正，保证后续处理方向一致
  private static func normalizedUp(_ image: UIImage) -> UIImage {
    guard image.imageOrientation != .up else { return image }
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    format.opaque = false
    let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
    return renderer.image { _ in
      image.draw(in: CGRect(origin: .zero, size: image.size))
    }
  }

  /// 最长边限制到 maxDim，控制 Vision 与内存开销
  private static func downscale(_ image: UIImage, maxDim: CGFloat) -> UIImage {
    let maxSide = max(image.size.width, image.size.height)
    guard maxSide > maxDim, maxSide > 0 else { return image }
    let ratio = maxDim / maxSide
    let newSize = CGSize(
      width: floor(image.size.width * ratio),
      height: floor(image.size.height * ratio)
    )
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    format.opaque = false
    return UIGraphicsImageRenderer(size: newSize, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: newSize))
    }
  }
}
