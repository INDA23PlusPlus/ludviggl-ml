
const std = @import("std");

const types  = @import("types.zig");
const Vector = types.Vector;
const Value  = types.Value;

pub fn ReLU(comptime N: usize, v: Vector(N)) Vector(N) {
    const zeroes: Vector(N) = @splat(0.0);
    const pred: @Vector(N, bool) = zeroes < v;
    return @select(Value, pred, v, zeroes);
}

pub fn softmax(comptime N: usize, v: Vector(N)) Vector(N) {
    var result: Vector(N) = undefined;
    for (0..N) |i| {
        result[i] = std.math.exp(v[i]);
    }
    const sum = @reduce(.Add, result);
    const sum_v: Vector(N) = @splat(sum);
    return result / sum_v;
}

test {

    const v: Vector(4) = .{ -1.0, 2.0, 3.0, -5.0, };
    const a = ReLU(4, v);

    try std.testing.expectEqual(a[0], 0.0);
    try std.testing.expectEqual(a[1], 2.0);
    try std.testing.expectEqual(a[2], 3.0);
    try std.testing.expectEqual(a[3], 0.0);
}
