
const Self = @This();

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

pub const PathSpec = struct {
    training_data:   [] const u8,
    training_labels: [] const u8,
    test_data:       [] const u8,
    test_labels:     [] const u8,
};

pub const sample_size = 28 * 28;

training_data:   ArrayList(f32),
training_labels: ArrayList(u8),
test_data:       ArrayList(f32),
test_labels:     ArrayList(u8),

pub fn init(path_spec: PathSpec, allocator: Allocator, max_size: usize) !Self {
    return .{
        .training_data   = try loadData(path_spec.training_data, allocator, max_size),
        .training_labels = try loadLabels(path_spec.training_labels, allocator, max_size),
        .test_data       = try loadData(path_spec.test_data, allocator, max_size),
        .test_labels     = try loadLabels(path_spec.test_labels, allocator, max_size),
    };
}

pub fn deinit(self: *Self) void {
    self.training_data.deinit();
    self.training_labels.deinit();
    self.test_data.deinit();
    self.test_labels.deinit();
}

pub fn trainingSize(self: *Self) usize {
    return self.training_labels.items.len;
}

pub fn testSize(self: *Self) usize {
    return self.test_labels.items.len;
}

pub fn imageSlice(data: *ArrayList(f32), index: usize) [] const f32 {
    const begin = index * sample_size;
    return data.items[begin..begin + sample_size];
}

fn loadData(path: [] const u8, allocator: Allocator, max_size: usize) !ArrayList(f32) {

    const file   = try std.fs.cwd().openFile(path, .{});
    var bufread  = std.io.bufferedReader(file.reader());
    const reader = bufread.reader();

    const magic = try reader.readIntBig(i32);
    if (magic != 0x803) return error.DarkMagic;

    const size = @min(max_size, @as(usize, @intCast(try reader.readIntBig(i32))));

    _ = try reader.readIntBig(i32); // rows
    _ = try reader.readIntBig(i32); // columns

    var data = ArrayList(f32).init(allocator);
    errdefer data.deinit();

    try data.ensureTotalCapacity(size * sample_size);

    for (0..size * sample_size) |i| {
        if (i % (sample_size * 1000) == 0) {
            std.debug.print("    {}/{} samples loaded.\r", .{ i / sample_size, size, });
        }
        const value: f32 = @floatFromInt(try reader.readIntBig(u8));
        data.appendAssumeCapacity(value / 255);
    }
    std.debug.print("    {0}/{0} samples loaded.\n", .{  size, });

    return data;
}

fn loadLabels(path: [] const u8, allocator: Allocator, max_size: usize) !ArrayList(u8) {

    const file = try std.fs.cwd().openFile(path, .{});
    const reader = file.reader();

    const magic = try reader.readIntBig(i32);
    if (magic != 0x801) return error.DarkMagic;

    const size = @min(max_size, @as(usize, @intCast(try reader.readIntBig(i32))));

    var data = ArrayList(u8).init(allocator);
    errdefer data.deinit();

    for (0..size) |_| {
        const value: u8 = try reader.readIntBig(u8);
        try data.append(value);
    }

    return data;
}
