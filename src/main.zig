
const std        = @import("std");
const Data       = @import("Data.zig");
const Network    = @import("network.zig").Network;
const types      = @import("types.zig");
const activation = @import("activation.zig");
const Value      = types.Value;
const Vector     = types.Vector;

//const palette = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

const layer_spec        = &[_] usize { 28 * 28, 16, 16, 10, };
const batch_size: usize = 50;
const epochs:     usize = 50;

fn digitVector(d: usize) Vector(10) {
    var result: Vector(10) = @splat(0);
    result[d] = 1;
    return result;
}

pub fn main() !void {

    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer _ = gpa.deinit();
    var allocator = gpa.allocator();

    const path_spec: Data.PathSpec = .{
        .training_data   = "data/train-images-idx3-ubyte",
        .training_labels = "data/train-labels-idx1-ubyte",
        .test_data       = "data/t10k-images-idx3-ubyte",
        .test_labels     = "data/t10k-labels-idx1-ubyte",
    };

    std.debug.print("Loading data set...\n", .{});
    var data = try Data.init(path_spec, allocator, 60000);
    defer data.deinit();

    std.debug.print("Training size: {d}, Test size: {d}\n", .{ data.trainingSize(), data.testSize(), });

    const layer_sizes = &[_] usize { 28 * 28, 16, 10, };
    var net = try Network(
        layer_sizes,
        activation.ReLU,
        activation.Softmax,
    ).init(allocator);
    defer net.deinit();

    var rng = std.rand.DefaultPrng.init(69);
    net.randomize(&rng, 0.3);

    var prev_cost: Value = std.math.inf(Value);

    std.debug.print("Training network...\n", .{});
    for (0..epochs) |e| {
        var cost: Value = 0;
        for (0..data.trainingSize() / batch_size) |b| {
            const b0 = b * batch_size;
            var b1 = (b + 1) * batch_size;
            if (b1 > data.trainingSize()) b1 = data.trainingSize();
            for (b0..b1) |i| {
                const in = Data.imageSlice(&data.training_data, i);
                const out = data.training_labels.items[i];
                const target = digitVector(out);
                try net.setInputSlice(in);
                net.forward();
                net.back(target);
                cost += net.cost(target);
            }
            net.descentGrad(0.05);
        }
        std.debug.print("\x1b[2K\rEpoch {}/{}", .{ e, epochs, });
        cost /= @as(Value, @floatFromInt(data.trainingSize()));

        if (cost > prev_cost and @abs(cost - prev_cost) > 0.001) {
            std.debug.print("\x1b[2K\r!!! Cost is increasing, aborting !!!\n", .{});
            break;
        }

        prev_cost = cost;

        std.debug.print(", cost = {d}", .{ cost, });

        const output = net.getOutput();
        std.debug.print(
            ", output: {d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}\r",
            .{
                output[1],
                output[2],
                output[3],
                output[4],
                output[5],
                output[6],
                output[7],
                output[8],
                output[9],
                output[10],
            }
        );
    }
    std.debug.print("\n", .{});

    std.debug.print("Testing...\n", .{});
    var correct: f32 = 0;
    for (0..data.testSize()) |i| {

        if (i % 100 == 0) {
            std.debug.print("\x1b[2K\rTested {}/{} samples", .{ i, data.testSize(), });
        }

        const img = Data.imageSlice(&data.test_data, i);
        try net.setInputSlice(img);
        net.forward();
        const output = net.getOutput().*;
        const expected: usize = @intCast(data.test_labels.items[i]);

        var result: usize = 0;
        var max: Value = 0;

        for (1..10) |j| {
            if (output[j] > max) {
                result = j - 1;
                max = output[j];
            }
        }

        if (result == expected) correct += 1;

    }
    std.debug.print("\x1b[2K\rTested {0}/{0} samples\n", .{ data.testSize(), });
    const fsize: f32 = @floatFromInt(data.testSize());
    const perc: f32 = 100 * correct / fsize;
    std.debug.print("Accuracy: {d:.2}%\n", .{ perc, });
}
