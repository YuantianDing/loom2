use std::ops::{Add, Neg, Shl, Shr, Sub};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct i6(u8);

impl From<u8> for i6 {
    fn from(value: u8) -> Self {
        assert!(value < 64);
        Self(value)
    }
}

impl Into<u8> for i6 {
    fn into(self) -> u8 {
        self.0
    }
}

impl From<i8> for i6 {
    fn from(value: i8) -> Self {
        assert!(value >= -32 && value < 32);
        if value < 0 {
            Self((value + 64) as u8)
        } else {
            Self(value as u8)
        }
    }
}

impl Into<i8> for i6 {
    fn into(self) -> i8 {
        if self.0 >> 5 == 0 {
            self.0 as i8
        } else {
            (self.0 as i8) - 64
        }
    }
}

impl Add for i6 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.0 + rhs.0;
        if sum < 64 {
            Self(sum)
        } else {
            Self(sum - 64)
        }
    }
}

impl Sub for i6 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(self.0 + 64 - rhs.0)
        }
    }
}

impl Neg for i6 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            self
        } else {
            Self(64 - self.0)
        }
    }
}

impl Shl<u8> for i6 {
    type Output = Self;

    fn shl(self, rhs: u8) -> Self::Output {
        assert!(rhs < 6);
        Self((self.0 << rhs) % 64)
    }
}

impl Shr<u8> for i6 {
    type Output = Self;

    fn shr(self, rhs: u8) -> Self::Output {
        assert!(rhs < 6);
        ((Into::<i8>::into(self) >> rhs) % 64).into()
    }
}

pub(crate) trait I6ShiftEncoding {
    fn base(level: u8) -> i6;
    fn level_perform(level: u8, action: i6, cur: i6) -> (bool, i6);
    fn encode(input: i6) -> i6;
    fn decode(action: i6) -> i6 {
        let mut diff: i6 = 0_u8.into();
        for level in 0..6 {
            diff = Self::level_perform(level, action, diff).1;
        }
        diff
    }
}

pub(crate) struct Negabinary;

impl I6ShiftEncoding for Negabinary {
    fn base(level: u8) -> i6 {
        if level % 2 == 0 {
            i6::from(1_u8 << level)
        } else {
            i6::from(64_u8 - (1_u8 << level))
        }
    }

    fn level_perform(level: u8, action: i6, cur: i6) -> (bool, i6) {
        let action = action;
        let action: u8 = action.into();
        if (action >> level) & 1 == 0 {
            (false, cur)
        } else {
            (true, cur + Self::base(level))
        }
    }

    fn encode(input: i6) -> i6 {
        let mut action = 0;
        let mut diff: i8 = input.into();
        for level in 0..6 {
            if diff == 0 {
                break;
            }
            let bit = (diff & 1) as u8;
            action |= bit << level;
            diff = (diff - bit as i8) / -2;
        }
        assert!(Self::decode(action.into()) == input);
        action.into()
    }
}

pub(crate) struct Binary;

impl I6ShiftEncoding for Binary {
    fn base(level: u8) -> i6 {
        assert!(level < 6);
        i6::from(1_u8 << level)
    }

    fn level_perform(level: u8, action: i6, cur: i6) -> (bool, i6) {
        let action: u8 = action.into();
        if (action >> level) & 1 == 0 {
            (false, cur)
        } else {
            (true, cur + Self::base(level))
        }
    }

    fn encode(input: i6) -> i6 {
        input
    }
}

pub(crate) struct BinaryBackward;

impl I6ShiftEncoding for BinaryBackward {
    fn base(level: u8) -> i6 {
        -Binary::base(level)
    }

    fn level_perform(level: u8, action: i6, cur: i6) -> (bool, i6) {
        let action: u8 = action.into();
        if (action >> level) & 1 == 0 {
            (false, cur)
        } else {
            (true, cur + Self::base(level))
        }
    }

    fn encode(input: i6) -> i6 {
        -input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_round_trips<E: I6ShiftEncoding>() {
        for raw in 0_u8..64 {
            let input = i6::from(raw);
            let action = E::encode(input);
            assert_eq!(E::decode(action), input, "failed for displacement {raw}");

            let mut current = i6::from(0_u8);
            for level in 0..6 {
                current = E::level_perform(level, action, current).1;
            }
            assert_eq!(current, input, "level execution failed for {raw}");
        }
    }

    #[test]
    fn binary_round_trips_every_six_bit_displacement() {
        assert_round_trips::<Binary>();
        for level in 0..6 {
            assert_eq!(Binary::base(level), i6::from(1_u8 << level));
        }
    }

    #[test]
    fn backward_binary_round_trips_every_six_bit_displacement() {
        assert_round_trips::<BinaryBackward>();
        for level in 0..6 {
            assert_eq!(BinaryBackward::base(level), -i6::from(1_u8 << level));
        }
    }

    #[test]
    fn negabinary_still_round_trips_every_six_bit_displacement() {
        assert_round_trips::<Negabinary>();
    }
}
