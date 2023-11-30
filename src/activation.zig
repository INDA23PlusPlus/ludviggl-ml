
const std = @import("std");

const Self = @This();

const types  = @import("types.zig");
const Vector = types.Vector;
const Value  = types.Value;

pub const ReLU = .{
    .act   = actReLU,
    .deriv = derivReLU,
};

pub const Softmax = .{
    .act   = actSoftmax,
    .deriv = derivSoftmax,
};

fn actReLU(comptime N: usize, v: Vector(N)) Vector(N) {
    const zeroes: Vector(N)        = @splat(0);
    const pred:   @Vector(N, bool) = zeroes < v;
    return @select(Value, pred, v, zeroes);
}

fn derivReLU(comptime N: usize, v: Vector(N)) Vector(N) {
    const zeroes: Vector(N)        = @splat(0);
    const ones:   Vector(N)        = @splat(1);
    const pred:   @Vector(N, bool) = zeroes < v;
    return @select(Value, pred, ones, zeroes);
}

fn actSoftmax(comptime N: usize, v: Vector(N)) Vector(N) {
    const exp = @exp(v);
    const sum: Vector(N) = @splat(@reduce(.Add, exp));
    return exp / sum;
}

fn derivSoftmax(comptime N: usize, v: Vector(N)) Vector(N) {
    const ones: Vector(N) = @splat(1);
    const sm = actSoftmax(N, v);
    return sm * (ones - sm);
}

test {

    const v: Vector(5) = .{ -1.0, 2.0, 3.0, -5.0, 11.0, };
    const a = actReLU(5, v);

    try std.testing.expectEqual(a[0], 0.0);
    try std.testing.expectEqual(a[1], 2.0);
    try std.testing.expectEqual(a[2], 3.0);
    try std.testing.expectEqual(a[3], 0.0);
    try std.testing.expectEqual(a[4], 10.0);
}
