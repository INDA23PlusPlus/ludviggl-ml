
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

pub fn init(path_spec: PathSpec, allocator: Allocator) !Self {
    return .{
        .training_data   = try loadData(path_spec.training_data, allocator),
        .training_labels = try loadLabels(path_spec.training_labels, allocator),
        .test_data       = try loadData(path_spec.test_data, allocator),
        .test_labels     = try loadLabels(path_spec.test_labels, allocator),
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

fn loadData(path: [] const u8, allocator: Allocator) !ArrayList(f32) {

    const file = try std.fs.cwd().openFile(path, .{});
    const reader = file.reader();

    const magic = try reader.readIntBig(i32);
    if (magic != 0x803) return error.DarkMagic;

    const size: usize = @intCast(try reader.readIntBig(i32));

    _ = try reader.readIntBig(i32); // rows
    _ = try reader.readIntBig(i32); // columns

    var data = ArrayList(f32).init(allocator);
    errdefer data.deinit();

    for (0..size) |_| {
        const value: f32 = @floatFromInt(try reader.readIntBig(u8));
        try data.append(value);
    }

    return data;
}

fn loadLabels(path: [] const u8, allocator: Allocator) !ArrayList(u8) {

    const file = try std.fs.cwd().openFile(path, .{});
    const reader = file.reader();

    const magic = try reader.readIntBig(i32);
    if (magic != 0x801) return error.DarkMagic;

    const size: usize = @intCast(try reader.readIntBig(i32));

    var data = ArrayList(u8).init(allocator);
    errdefer data.deinit();

    for (0..size) |_| {
        const value: u8 = try reader.readIntBig(u8);
        try data.append(value);
    }

    return data;
}
