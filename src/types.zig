
pub const Value = f32;

pub fn Vector(comptime N: usize) type {
    return @Vector(N, Value);
}

pub fn Matrix(comptime M: usize, comptime N: usize) type {
    return struct {

        const Self = @This();

        const rows = N;
        const cols = M;

        data: [N] Vector(M),

        pub fn init(data: [N] Vector(N)) Self {
            return .{ .data = data, };
        }

        pub fn mul(self: Self, vec: Vector(M)) Vector(N) {
            var result: Vector(N) = undefined;
            for (0..N) |i| result[i] = @reduce(.Add, vec * self.data[i]);
            return result;
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

    const std = @import("std");

    var mat = Matrix(3, 3).init(
        .{
            .{  3, 4, -1, },
            .{  1, 1,  0, },
            .{ -1, 2,  3, },
        }
    );

    const vec: Vector(3) = .{ 1, -1, -2, };

    const result = mat.mul(vec);

    try std.testing.expectEqual(result[0],  1);
    try std.testing.expectEqual(result[1],  0);
    try std.testing.expectEqual(result[2], -9);
}
