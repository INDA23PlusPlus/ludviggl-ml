
const std       = @import("std");
const simd      = std.simd;
const Allocator = std.mem.Allocator;
const Arena     = std.heap.ArenaAllocator;

const types     = @import("types.zig");
const Value     = types.Value;
const Matrix    = types.Matrix;
const Vector    = types.Vector;

const unit: Vector(1) = .{ 1, };

pub fn Network(
    comptime LayerSpec:        [] const usize,
    comptime MiddleActivation: anytype,
    comptime OutputActivation: anytype
) type {

    return struct {

        const Self = @This();

        const A = MiddleActivation;
        const O = OutputActivation;
        const C = LayerSpec.len;

        // Increase each layer size by one
        //  to account for bias
        const S = blk: {
            var aug: [C] usize = undefined;
            for (LayerSpec, &aug) |l, *a| {
                a.* = l + 1;
            }
            break :blk aug;
        };

        /// Layer outputs
        outs:    [C] *anyopaque,
        /// Layer activations
        acts:    [C] *anyopaque,
        /// Layer costs
        costs:   [C] *anyopaque,
        /// Weights
        weights: [C] *anyopaque,
        /// Gradient
        grads:   [C] *anyopaque,

        arena:       Arena,
        batch_scale: Value = 0,

        pub fn init(allocator: Allocator) !Self {

            var self: Self = undefined;
            self.arena = Arena.init(allocator);

            inline for (0..C) |i| {
                const s = S[i];
                self.outs[i]  = try self.arena.allocator().create(Vector(s));
            }

            inline for (1..C) |i| {
                const s = S[i];
                const p = S[i - 1];

                self.acts[i]    = try self.arena.allocator().create(Vector(s));
                self.costs[i]   = try self.arena.allocator().create(Vector(s));
                self.weights[i] = try self.arena.allocator().create(Matrix(p, s));
                self.grads[i]   = try self.arena.allocator().create(Matrix(p, s));
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
        }

        pub fn setInput(self: *Self, in: Vector(S[0] - 1)) void {
            self.getOuts(0).* = simd.join(unit, in);
        }

        pub fn setInputSlice(self: *Self, in: [] const Value) !void {
            if (in.len != S[0] - 1) return error.LayerSizeMisMatch;
            const outs = self.getOuts(0);
            outs[0] = 1;
            for (in, 1..) |v, i| {
                outs[i] = v;
            }
        }

        pub fn forward(self: *Self) void {
            inline for (1..C) |i| {
                self.forwardLayer(i);
            }
        }

        pub fn back(self: *Self, target: Vector(S[C - 1] - 1)) void {
            self.compCosts(target);
            self.accumGrads();
            self.batch_scale += 1;
        }

        pub fn cost(self: *Self, target: Vector(S[C - 1] - 1)) Value {
            const diff = simd.join(unit, target) - self.getOuts(C - 1).*;
            const half: Vector(S[C - 1]) = @splat(0.5);
            return @reduce(.Add, diff * diff * half);
        }

        pub fn randomize(self: *Self, rng: anytype, f: Value) void {
            inline for (1..C) |i| {
                const weights = self.getWeights(i);
                for (&weights.data) |*r| {
                    for (0..S[i - 1]) |c| {
                        r[c] = f * (2 * rng.random().float(Value) - 1);
                    }
                }
            }
        }

        pub fn getOutput(self: *Self) *Vector(S[C - 1]) {
            return self.getOuts(C - 1);
        }

        fn forwardLayer(self: *Self, comptime i: usize) void {
            const in      = self.getOuts(i - 1);
            const weights = self.getWeights(i);
            const acts    = self.getActs(i);
            const outs    = self.getOuts(i);
            acts.*        = weights.mul(in.*);
            const actFn   = if (i < C - 1) A.act else O.act;
            outs.*        = actFn(S[i], acts.*);
            outs[0]       = 1; // bias
        }

        fn compCosts(self: *Self, target: Vector(S[C - 1] - 1)) void {
            self.compCostsOut(target);
            inline for (1..C - 1) |i| {
                self.compCostsMiddle(C - i - 1);
            }
        }

        fn compCostsOut(self: *Self, target: Vector(S[C - 1] - 1)) void {
            const outs  = self.getOuts(C - 1);
            const acts  = self.getActs(C - 1);
            const costs = self.getCosts(C - 1);
            const deriv = O.deriv(S[C - 1], acts.*);
            const diff  = simd.join(unit, target) - outs.*;
            costs.* = deriv * diff;
            costs[0] = 0;
        }

        fn compCostsMiddle(self: *Self, comptime i: usize) void {
            const next_costs   = self.getCosts(i + 1);
            const next_weights = self.getWeights(i + 1);
            const this_acts    = self.getActs(i);
            const this_costs   = self.getCosts(i);
            const s            = A.deriv(S[i], this_acts.*);
            const v            = next_weights.mul_transpose(next_costs.*);
            this_costs.* = s * v;
        }

        fn accumGrads(self: *Self) void {
            inline for (1..C) |i| {
                self.accumGradsLayer(i);
            }
        }

        fn accumGradsLayer(self: *Self, comptime i: usize) void {
            const costs   = self.getCosts(i);
            const outs    = self.getOuts(i - 1);
            const grads   = self.getGrads(i);
            for (0..S[i]) |j| {
                const s: Vector(S[i - 1]) = @splat(costs[j]);
                grads.data[j] += outs.* * s;
            }
        }

        pub fn descentGrad(self: *Self, rate: Value) void {
            inline for (1..C) |i| {
                var g = self.getGrads(i).scale(rate / self.batch_scale);
                g.clamp(1);
                self.getWeights(i).acc(g);
            }
            self.resetGrad();
            self.batch_scale = 0;
        }

        fn resetGrad(self: *Self) void {
            inline for (1..C) |i| {
                self.getGrads(i).zero();
            }
        }

        fn getOuts(self: Self, comptime i: usize) *Vector(S[i]) {
            return @alignCast( @ptrCast(self.outs[i]) );
        }

        fn getActs(self: Self, comptime i: usize) *Vector(S[i]) {
            return @alignCast( @ptrCast(self.acts[i]) );
        }

        fn getCosts(self: Self, comptime i: usize) *Vector(S[i]) {
            return @alignCast( @ptrCast(self.costs[i]) );
        }

        fn getWeights(self: Self, comptime i: usize) *Matrix(S[i - 1], S[i]) {
            return @alignCast( @ptrCast(self.weights[i]) );
        }

        fn getGrads(self: Self, comptime i: usize) *Matrix(S[i - 1], S[i]) {
            return @alignCast( @ptrCast(self.grads[i]) );
        }
    };
}
