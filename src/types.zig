
const std = @import("std");

pub const Value = f32;

pub fn Vector(comptime N: usize) type {
    return @Vector(N, Value);
}

pub fn Matrix(comptime M: usize, comptime N: usize) type {

    return struct {

        const Self = @This();

        pub const rows = N;
        pub const cols = M;

        data: [N] Vector(M),

        pub fn init(data: [N] Vector(N)) Self {
            return .{ .data = data, };
        }

        pub fn mul(self: Self, vec: Vector(M)) Vector(N) {
            var result: Vector(N) = undefined;
            for (0..N) |i| result[i] = @reduce(.Add, vec * self.data[i]);
            return result;
        }

        pub fn mul_transpose(self: Self, vec: Vector(N)) Vector(M) {
            var result: Vector(M) = @splat(0);
            for (0..N) |i| {
                result += @as(Vector(M), @splat(vec[i])) * self.data[i];
            }
            return result;
        }

        pub fn acc(self: *Self, other: Self) void {
            for (0..N) |i| self.data[i] += other.data[i];
        }

        pub fn zero(self: *Self) void {
            for (&self.data) |*r| r.* = @as(Vector(M), @splat(0));
        }

        pub fn scale(self: Self, s: Value) Self {
            var result: Self      = undefined;
            const sv:   Vector(M) = @splat(s);

            for (self.data, &result.data) |d, *r| {
                r.* = d * sv;
            }

            return result;
        }

        pub fn clamp(self: *Self, range: Value) void {
            const min: Vector(M) = @splat(-range);
            const max: Vector(M) = @splat(range);
            for (&self.data) |*row| {
                for (0..M) |i| {
                    if (std.math.isNan(row[i]) or std.math.isInf(row[i])) row[i] = 0;
                }
                const minpred = row.* < min;
                const maxpred = row.* > max;
                row.* = @select(Value, minpred, min, row.*);
                row.* = @select(Value, maxpred, max, row.*);
            }
        }

        pub fn set(self: *Self, r: usize, c: usize, v: Value) void {
            self.data[r][c] = v;
        }

        pub fn get(self: *Self, r: usize, c: usize) Value {
            return self.data[r][c];
        }
    };
}

test {

    var mat = Matrix(3, 3).init(
        .{
            .{  3, 4, -1, },
            .{  1, 1,  0, },
            .{ -1, 2,  3, },
        }
    );

    const vec: Vector(3) = .{ 1, -1, -2, };

    var result = mat.mul(vec);
    try std.testing.expectEqual(result, Vector(3) { 1, 0, -9, });

    result = mat.mul_transpose(vec);
    try std.testing.expectEqual(result, Vector(3) { 4, -1, -7, });

    mat.clamp(1);
    try std.testing.expectEqual(mat.data[0], Vector(3) {  1,  1, -1, });
    try std.testing.expectEqual(mat.data[1], Vector(3) {  1,  1,  0, });
    try std.testing.expectEqual(mat.data[2], Vector(3) { -1,  1,  1, });
}
