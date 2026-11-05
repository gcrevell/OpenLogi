//! SF Symbols, rasterised into an alpha mask the ring tints and draws.
//!
//! GPUI draws bundled SVGs and bitmaps, not system symbols, so the symbol is
//! drawn once into an `NSBitmapImageRep` and only its alpha channel is kept:
//! the ring owns the glyph colour (it changes with the hovered slot and the
//! appearance), and a mask can be recoloured without asking AppKit again.

/// A symbol's coverage, one byte per pixel, in a square `edge` × `edge` canvas
/// with the symbol centred and scaled to fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SymbolMask {
    pub edge: u32,
    pub alpha: Vec<u8>,
}

/// A 0–1 intensity as a byte.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 0..=255 before the cast"
)]
pub(crate) fn unit_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl SymbolMask {
    /// The mask painted in `color` (straight alpha, RGBA), as the BGRA texture
    /// GPUI draws. Returns `None` if the buffer does not fit its edge.
    pub(crate) fn tinted(&self, color: gpui::Rgba) -> Option<std::sync::Arc<gpui::RenderImage>> {
        let (red, green, blue) = (
            unit_to_byte(color.r),
            unit_to_byte(color.g),
            unit_to_byte(color.b),
        );
        let bgra = self
            .alpha
            .iter()
            .flat_map(|&coverage| {
                let alpha = unit_to_byte(f32::from(coverage) / 255.0 * color.a);
                [blue, green, red, alpha]
            })
            .collect();
        let buffer = image::RgbaImage::from_raw(self.edge, self.edge, bgra)?;
        Some(std::sync::Arc::new(gpui::RenderImage::new(vec![
            image::Frame::new(buffer),
        ])))
    }
}

/// Rasterise the SF Symbol `name`, or `None` when this system has no such
/// symbol or cannot draw it.
#[cfg(target_os = "macos")]
#[expect(
    unsafe_code,
    reason = "NSBitmapImageRep's designated initialiser takes a nullable plane pointer, and NSDeviceRGBColorSpace is an extern static; both are plain AppKit FFI"
)]
pub(crate) fn rasterize(name: &str, edge: u32) -> Option<SymbolMask> {
    use objc2::AnyThread as _;
    use objc2::rc::Retained;
    use objc2_app_kit::{
        NSBitmapImageRep, NSCompositingOperation, NSDeviceRGBColorSpace, NSGraphicsContext, NSImage,
    };
    use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

    if edge == 0 {
        return None;
    }
    let image = NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str(name),
        None,
    )?;
    let natural = image.size();
    if natural.width <= 0.0 || natural.height <= 0.0 {
        return None;
    }
    let side = isize::try_from(edge).ok()?;
    // SAFETY: a null plane pointer asks AppKit to allocate and own the pixel
    // buffer; the remaining arguments describe a plain 8-bit RGBA bitmap.
    let rep: Retained<NSBitmapImageRep> = unsafe {
        NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
            NSBitmapImageRep::alloc(),
            std::ptr::null_mut(),
            side,
            side,
            8,
            4,
            true,
            false,
            // SAFETY: `NSDeviceRGBColorSpace` is an immutable AppKit constant.
            NSDeviceRGBColorSpace,
            side * 4,
            32,
        )
    }?;
    let data = rep.bitmapData();
    if data.is_null() {
        return None;
    }
    let len = usize::try_from(edge)
        .ok()?
        .checked_mul(usize::try_from(edge).ok()?)?
        * 4;
    // SAFETY: AppKit allocated `bytesPerRow * pixelsHigh` = `len` bytes for
    // `rep`, which stays alive (and unresized) for this whole function.
    let pixels = unsafe { std::slice::from_raw_parts_mut(data, len) };
    pixels.fill(0);

    let context = NSGraphicsContext::graphicsContextWithBitmapImageRep(&rep)?;
    let canvas = f64::from(edge);
    let scale = (canvas / natural.width).min(canvas / natural.height);
    let (width, height) = (natural.width * scale, natural.height * scale);
    let target = NSRect::new(
        NSPoint::new((canvas - width) / 2.0, (canvas - height) / 2.0),
        NSSize::new(width, height),
    );
    NSGraphicsContext::saveGraphicsState_class();
    NSGraphicsContext::setCurrentContext(Some(&context));
    image.drawInRect_fromRect_operation_fraction(
        target,
        NSRect::ZERO,
        NSCompositingOperation::SourceOver,
        1.0,
    );
    context.flushGraphics();
    NSGraphicsContext::restoreGraphicsState_class();

    // SAFETY: as above; drawing has finished, so nothing else touches the buffer.
    let pixels = unsafe { std::slice::from_raw_parts(data, len) };
    let alpha: Vec<u8> = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| pixel[3])
        .collect();
    alpha
        .iter()
        .any(|&a| a != 0)
        .then_some(SymbolMask { edge, alpha })
}

/// Only macOS has SF Symbols; the ring keeps its bundled icon.
#[cfg(not(target_os = "macos"))]
pub(crate) fn rasterize(_name: &str, _edge: u32) -> Option<SymbolMask> {
    None
}

#[cfg(test)]
mod tint_tests {
    use super::SymbolMask;

    #[test]
    fn tinting_keeps_coverage_as_alpha_and_orders_channels_bgra() {
        let mask = SymbolMask {
            edge: 2,
            alpha: vec![0, 255, 128, 255],
        };
        let image = mask
            .tinted(gpui::Rgba {
                r: 1.0,
                g: 0.5,
                b: 0.0,
                a: 1.0,
            })
            .expect("a 2x2 mask fits a 2x2 image");
        let bytes = image.as_bytes(0).expect("one frame");
        assert_eq!(&bytes[0..4], [0, 128, 255, 0]);
        assert_eq!(&bytes[4..8], [0, 128, 255, 255]);
        assert_eq!(bytes[11], 128);
    }

    #[test]
    fn a_translucent_colour_scales_coverage() {
        let mask = SymbolMask {
            edge: 1,
            alpha: vec![200],
        };
        let image = mask
            .tinted(gpui::Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.5,
            })
            .expect("a 1x1 mask fits a 1x1 image");
        assert_eq!(image.as_bytes(0).expect("one frame")[3], 100);
    }
}

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::rasterize;

    #[test]
    fn a_known_symbol_draws_something_inside_its_canvas() {
        let mask = rasterize("play.fill", 48).expect("play.fill ships with macOS");
        assert_eq!(mask.alpha.len(), 48 * 48);
        let covered = mask.alpha.iter().filter(|&&a| a > 0).count();
        assert!(covered > 100, "only {covered} pixels covered");
        assert!(
            covered < 48 * 48,
            "the symbol must not fill its whole canvas"
        );
    }

    #[test]
    fn an_unknown_symbol_is_none() {
        assert_eq!(rasterize("definitely.not.a.symbol", 48), None);
    }

    #[test]
    fn a_zero_sized_canvas_is_none() {
        assert_eq!(rasterize("play.fill", 0), None);
    }
}
