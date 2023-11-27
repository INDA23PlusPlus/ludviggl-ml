
const std       = @import("std");
const Allocator = std.mem.Allocator;
const Arena     = std.heap.ArenaAllocator;

const types  = @import("types.zig");
const Vector = types.Vector;
const Matrix = types.Matrix;
const Value  = types.Value;

pub fn Network(comptime L: [] const usize, comptime A: anytype) type {

    return struct {

        const Self = @This();

        layers:  [L.len] *anyopaque = undefined,
        weights: [L.len] *anyopaque = undefined,
        arena:   Arena,

        pub fn init(allocator: Allocator) !Self {

            var self: Self = undefined;
            self.arena = Arena.init(allocator);
            errdefer _ = self.arena.reset(.free_all);

            inline for (L, 0..) |s, i| {
                self.layers[i] = try self.arena.allocator().create(Vector(s));
            }

            inline for (1..L.len) |i| {
                self.weights[i] = try self.arena.allocator().create(Matrix(L[i - 1] + 1, L[i]));
            }

            return self;
        }

        pub fn deinit(self: *Self) bool {
            return self.arena.reset(.free_all);
        }

        pub fn forward(self: *Self) void {

            inline for (1..L.len) |i| {

                const in = self.getLayer(i - 1);
                const bias: Vector(1) = .{ 1, };
                const aug = std.simd.join(bias, in.*);

                const out = self.getLayer(i);
                const weights = self.getWeights(i);

                out.* = A(L[i], weights.mul(aug));
            }
        }

        fn getLayer(self: *Self, comptime N: usize) *Vector(L[N]) {
            return @alignCast( @ptrCast(self.layers[N]) );
        }

        fn getWeights(self: *Self, comptime N: usize) *Matrix(L[N - 1] + 1, L[N]) {
            return @alignCast( @ptrCast(self.weights[N]) );
        }
    };
}
