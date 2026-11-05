//! A validated RGB color for the lighting config.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An RGB color, parsed once at the boundary from the config/UI hex form:
/// `"RRGGBB"`, optionally with a leading `#`, or the `"RGB"` shorthand.
///
/// Always serializes as bare lowercase `"rrggbb"`, so the type is drop-in TOML- and
/// wire-compatible with the raw `String` field it replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

impl Rgb {
    /// White — the lighting default.
    pub const WHITE: Self = Self::new(0xff, 0xff, 0xff);

    /// A color from its red/green/blue components.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// The `(r, g, b)` components.
    #[must_use]
    pub const fn components(self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }

    /// The color packed as `0xRRGGBB` (the form GPUI's `rgb()` takes).
    #[must_use]
    pub const fn packed(self) -> u32 {
        (self.r as u32) << 16 | (self.g as u32) << 8 | self.b as u32
    }
}

/// A color string that is not 3 or 6 hex digits, optionally after a `#`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid RGB color {input:?}: expected hex digits (\"RRGGBB\" or \"RGB\", optional '#')")]
pub struct RgbParseError {
    /// The rejected input.
    input: String,
}

impl FromStr for Rgb {
    type Err = RgbParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let digits = s.strip_prefix('#').unwrap_or(s);
        let invalid = || RgbParseError { input: s.into() };
        if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid());
        }
        let packed = match digits.len() {
            6 => u32::from_str_radix(digits, 16).map_err(|_| invalid())?,
            3 => {
                let short = u32::from_str_radix(digits, 16).map_err(|_| invalid())?;
                let (r, g, b) = (short >> 8 & 0xf, short >> 4 & 0xf, short & 0xf);
                ((r * 0x11) << 16) | ((g * 0x11) << 8) | (b * 0x11)
            }
            _ => return Err(invalid()),
        };
        Ok(Self::new(
            (packed >> 16 & 0xff) as u8,
            (packed >> 8 & 0xff) as u8,
            (packed & 0xff) as u8,
        ))
    }
}

impl TryFrom<String> for Rgb {
    type Error = RgbParseError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<Rgb> for String {
    fn from(color: Rgb) -> Self {
        color.to_string()
    }
}

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::Rgb;

    #[test]
    fn parses_and_round_trips_hex() {
        let color: Rgb = "8000ff".parse().unwrap();
        assert_eq!(color, Rgb::new(0x80, 0x00, 0xff));
        assert_eq!(color.packed(), 0x0080_00ff);
        assert_eq!(color.to_string(), "8000ff");
    }

    #[test]
    fn accepts_uppercase_but_prints_lowercase() {
        let color: Rgb = "FF3B30".parse().unwrap();
        assert_eq!(color.to_string(), "ff3b30");
    }

    #[test]
    fn accepts_hash_prefix_and_three_digit_shorthand() {
        for (input, expected) in [
            ("#AE50FE", Rgb::new(0xae, 0x50, 0xfe)),
            ("#abc", Rgb::new(0xaa, 0xbb, 0xcc)),
            ("F0a", Rgb::new(0xff, 0x00, 0xaa)),
        ] {
            assert_eq!(input.parse::<Rgb>().unwrap(), expected, "{input:?}");
        }
        assert_eq!("#ABC".parse::<Rgb>().unwrap().to_string(), "aabbcc");
    }

    #[test]
    fn rejects_wrong_length_double_prefix_and_non_hex() {
        for bad in [
            "ff00aa0", "ffff", "##ff00aa", "#", "red", "", "+f00aa", "#ggg",
        ] {
            assert!(bad.parse::<Rgb>().is_err(), "{bad:?} should not parse");
        }
    }
}
