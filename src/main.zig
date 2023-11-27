
const std        = @import("std");
const Data       = @import("Data.zig");
const Network    = @import("network.zig").Network;
const types      = @import("types.zig");
const activation = @import("activation.zig");
const Value      = types.Value;

const palette = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

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

    var data = try Data.init(path_spec, allocator);
    defer data.deinit();

    std.debug.print("Training size: {d}, Test size: {d}\n", .{ data.trainingSize(), data.testSize(), });

    const test_id: usize = 2;
    const image = Data.imageSlice(&data.training_data, test_id);

    for (0..28) |row| {
        for (0..28) |col| {
            const palette_id: usize = palette.len - 1 - @as(usize, @intFromFloat(palette.len * image[col + 28 * row] / 255.99));
            const pixel: u8 = palette[palette_id];
            std.debug.print("{c}", .{ pixel, });
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\nYou should be seeing a {d}\n", .{ data.training_labels.items[test_id], });

    const layer_sizes = &[_] usize { 28 * 28, 5, 5, 10, };
    var net = try Network(layer_sizes, activation.ReLU, activation.softmax).init(allocator);
    defer _ = net.deinit();

    var rng = std.rand.DefaultPrng.init(69);
    net.randomize(&rng);

    try net.setInput(Data.imageSlice(&data.training_data, 0));

    net.forward();
    const err = net.err(.{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, });
    std.debug.print("Error: {e}\n", .{ err, });
}
