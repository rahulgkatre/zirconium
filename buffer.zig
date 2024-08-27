const std = @import("std");
const loop = @import("loop.zig");

fn ShapeToArray(comptime dtype: type, comptime ndims: u8, comptime shape: [ndims]usize) type {
    var Type = dtype;
    for (0..shape.len) |dim| {
        const s = shape[ndims - dim - 1];
        Type = [s]Type;
    }
    return Type;
}

pub const Layout = struct {
    original_ndims: *const u8,
    original_shape: []const usize,
    ndims: u8,
    shape: []const usize,
    strides: []usize,
    offest: usize = 0,
    skew: []usize,
    vectorized: bool = false,
    unrolled: std.DynamicBitSet,
    parallelized: std.DynamicBitSet,
};

pub fn Buffer(comptime Array: type) type {
    return struct {
        const Self = @This();
        multi: *Array,
        data: [*]dtype,
        layout: Layout,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const slice = try allocator.alloc(dtype, t_numel.?);

            if (dtype == bool) {
                @memset(slice, false);
            } else {
                @memset(slice, 0);
            }
            return .{
                .allocator = allocator,
                .multi = @ptrCast(slice.ptr),
                .data = @ptrCast(slice.ptr),
                .layout = .{
                    .original_ndims = &@intCast(t_ndims),
                    .original_shape = &t_shape,
                    .ndims = t_ndims,
                    .shape = &t_shape,
                    .strides = blk: {
                        const s = try allocator.alloc(usize, t_ndims);
                        @memcpy(s, &t_contiguous_strides);
                        break :blk s;
                    },
                    .skew = blk: {
                        const s = try allocator.alloc(usize, t_ndims);
                        @memset(s, 0);
                        break :blk s;
                    },
                    .unrolled = try std.DynamicBitSet.initEmpty(allocator, t_ndims),
                    .parallelized = try std.DynamicBitSet.initEmpty(allocator, t_ndims),
                },
            };
        }

        inline fn unravel(b: *const Self, idx: []const usize) usize {
            std.debug.assert(idx.len == b.layout.original_ndims.*);
            var i: usize = 0;
            inline for (0..t_ndims) |d| {
                comptime if (Base != dtype and d == t_ndims - 1) break;
                i += idx[d] * b.layout.strides[d];
            }
            return i;
        }

        pub inline fn load(b: *const Self, idx: []const usize) Base {
            if (comptime Base != dtype) {
                return @as(Base, b.data[b.unravel(idx)..][0..t_shape[t_ndims - 1]].*);
            } else {
                return b.data[b.unravel(idx)];
            }
        }

        pub inline fn store(b: *Self, data: Base, idx: []const usize) void {
            if (comptime Base != dtype) {
                b.data[b.unravel(idx)..][0..t_shape[t_ndims - 1]].* = data;
            } else {
                b.data[b.unravel(idx)] = data;
            }
        }

        pub const dtype: type = blk: {
            switch (@typeInfo(Array)) {
                .Vector => |info| break :blk info.child,
                .Pointer => |info| if (info.size == .Many) break :blk Buffer(info.child).dtype,
                .Array => |info| break :blk Buffer(info.child).dtype,
                .Int, .Float, .Bool => break :blk Array,
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        const t_ndims = blk: {
            switch (@typeInfo(Array)) {
                .Pointer => |info| if (info.size == .Many) break :blk 1 + Buffer(info.child).t_ndims,
                .Array => |info| break :blk 1 + Buffer(info.child).t_ndims,
                .Int, .Float, .Bool => break :blk 0,
                .Vector => break :blk 1,
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        const t_shape: [t_ndims]usize = blk: {
            switch (@typeInfo(Array)) {
                .Pointer => |info| if (info.size == .Many) break :blk .{null} ++ Buffer(info.child).t_shape,
                .Array => |info| break :blk .{info.len} ++ Buffer(info.child).t_shape,
                .Int, .Float, .Bool => break :blk .{},
                .Vector => |info| {
                    std.debug.assert(Buffer(info.child).dtype == dtype);
                    break :blk .{info.len};
                },
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        const t_contiguous_strides: [t_ndims]usize = blk: {
            if (t_ndims == 0) {
                break :blk .{};
            }
            var offset: usize = 1;
            var strides: [t_ndims]usize = undefined;
            for (0..t_ndims - 1) |d| {
                const stride = t_shape[t_ndims - d - 1] * offset;
                strides[t_ndims - d - 2] = stride;
                offset = stride;
            }
            strides[t_ndims - 1] = 1;
            for (0..t_ndims) |d| {
                if (t_shape[d] == 0 or t_shape[d] == 1) {
                    strides[d] = 0;
                }
            }
            break :blk strides;
        };

        const t_numel: ?comptime_int = blk: {
            var n = 1;
            for (t_shape) |len| {
                n *= len;
            }
            break :blk n;
        };

        pub const Flat: type = if (t_numel) |n| [n]dtype else [*]dtype; // if not symbolic, otherwise it is just [*]dtype

        pub const Base: type = blk: {
            switch (@typeInfo(Array)) {
                .Pointer => |info| if (info.size == .Many) break :blk Buffer(info.child).Base,
                .Array => |info| break :blk Buffer(info.child).Base,
                .Int, .Float, .Bool => break :blk dtype,
                .Vector => break :blk @Vector(t_shape[t_ndims - 1], dtype),
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        fn Split(comptime dim: u8, comptime count: u8) type {
            std.debug.assert(0 >= dim and dim < t_ndims);
            // TODO: Support splits that don't divide evenly
            // Give the option of how to evaluate the uneven part
            // - Pad it to evenly divide
            // - Unfuse into two separate loops (gives more control for unrolling)
            std.debug.assert(@mod(t_shape[dim], count) == 0);
            const pre = if (dim > 0) t_shape[0..dim].* else .{};
            const post = if (dim < t_ndims - 1) t_shape[dim + 1 .. t_ndims].* else .{};
            return Buffer(ShapeToArray(dtype, t_ndims + 1, pre ++ .{ @divExact(t_shape[dim], count), count } ++ post));
        }

        pub fn split(b: *Self, comptime dim: u8, comptime count: u8) *Split(dim, count) {
            b.layout.ndims += 1;
            b.layout.shape = &Split(dim, count).t_shape;

            const pre: [dim]usize = if (dim > 0) b.layout.strides[0..dim].* else .{};
            const new: [2]usize = .{ b.layout.strides[dim] * count, b.layout.strides[dim] };
            const post: [t_ndims - dim - 1]usize = if (dim < t_ndims - 1) b.layout.strides[dim + 1 .. t_ndims].* else .{};
            b.layout.strides = b.allocator.realloc(b.layout.strides, t_ndims + 1) catch unreachable;
            @memcpy(b.layout.strides[0..dim], &pre);
            @memcpy(b.layout.strides[dim .. dim + 2], &new);
            @memcpy(b.layout.strides[dim + 2 .. t_ndims + 1], &post);
            return @ptrCast(b);
        }

        pub fn constant(_: *const Self, val: dtype) Base {
            if (Base == dtype) {
                return val;
            } else {
                return @splat(val);
            }
        }

        /// Vectorized contiguous dim of the buffer.
        pub const Vectorized: type = t: {
            // do not vectorize a buffer that is already vectorized.
            if (Base != dtype) {
                break :t void;
            }
            var Type: type = @Vector(t_shape[t_ndims - 1], dtype);
            for (0..t_ndims - 1) |dim| {
                Type = [t_shape[t_ndims - dim - 2]]Type;
            }
            break :t Buffer(Type);
        };

        const NonVectorized = if (Vectorized != void) Self else void;

        pub fn vectorize(b: *NonVectorized) *Vectorized {
            std.debug.assert(b.layout.strides[b.layout.ndims - 1] == 1);
            b.layout.vectorized = true;
            return @ptrCast(b);
        }

        fn buildLoop(b: *const Self, dim: u8) !?*loop.Loop {
            if (dim >= b.layout.ndims) {
                return null;
            }
            if (b.layout.vectorized and dim == b.layout.ndims - 1) {
                return null;
            } else {
                const new_loop = try b.allocator.create(loop.Loop);
                new_loop.* = .{
                    .lower = b.layout.skew[dim],
                    .upper = b.layout.shape[dim],
                };
                return new_loop;
            }
        }

        pub fn nest(b: *const Self) !loop.Nest {
            if (b.layout.ndims == 0) {
                @panic("cannot generate loop nest for 0 dimensional buffer");
            }
            const top = try b.buildLoop(0);
            var curr = top;

            for (1..b.layout.ndims) |dim| {
                if (curr) |curr_loop| {
                    const inner = try b.buildLoop(@intCast(dim));
                    if (inner) |inner_loop| {
                        curr_loop.inner = inner_loop;
                    }
                    curr = inner;
                } else {
                    break;
                }
            }

            return .{
                .loop = top,
            };
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var b = try Buffer([16][8]f32).init(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&b.multi[0]) == @intFromPtr(&b.data[0]));
    try std.testing.expect(@intFromPtr(&b.multi[1][7]) == @intFromPtr(&b.data[15]));
}

test "split" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var b = try Buffer([16][8]f32).init(arena.allocator());
    defer arena.deinit();

    const sb = b.split(0, 4);
    try std.testing.expect(@TypeOf(sb) == *Buffer([4][4][8]f32));
    try std.testing.expectEqualSlices(usize, &.{ 32, 8, 1 }, b.layout.strides);
}

test "vectorize" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var b = try Buffer([16][8]f32).init(arena.allocator());
    defer arena.deinit();

    const vb = b.vectorize();
    try std.testing.expect(@TypeOf(vb) == *Buffer([16]@Vector(8, f32)));
}

test "split_vectorize" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var b = try Buffer([16][8]f32).init(arena.allocator());
    defer arena.deinit();

    const svb = b
        .split(0, 4)
        .vectorize();
    try std.testing.expect(@TypeOf(svb) == *Buffer([4][4]@Vector(8, f32)));
    try std.testing.expectEqualSlices(usize, &.{ 32, 8, 1 }, svb.layout.strides);
}

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const expected = comptime &loop.Loop{
        .upper = 16,
        .inner = &.{
            .upper = 8,
            .inner = null,
        },
    };
    var b = try Buffer([16][8]f32).init(arena.allocator());
    defer arena.deinit();

    const nest = try b.nest();
    try std.testing.expectEqualDeep(expected, nest.loop);
}

test "vectorize_nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const expected = comptime &loop.Loop{
        .upper = 16,
        .inner = null,
    };
    var b = try Buffer([16][8]f32).init(arena.allocator());
    defer arena.deinit();

    const nest = try b.vectorize().nest();
    try std.testing.expectEqualDeep(expected, nest.loop);
}

test "nest_eval" {
    const B = Buffer([16][8]i32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var b = try B.init(arena.allocator());
    defer arena.deinit();

    const nest = try b.nest();
    const Args = struct {
        b: B,
    };

    const logic: loop.Logic(Args, Args) = struct {
        fn logic(b1: *const B, b2: *B, idx: []const usize) void {
            const val = b1.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    nest.eval(Args, Args, 2, .{&b}, .{&b}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** 128), b.data[0..128]);
}

test "vectorize_nest_eval" {
    const B = Buffer([16][8]i32);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var b = try B.init(arena.allocator());
    defer arena.deinit();

    const vb = b.vectorize();
    const VB = @TypeOf(vb.*);
    try std.testing.expect(VB.Base == @Vector(8, i32));
    const nest = try vb.nest();
    const Args = struct {
        b: VB,
    };

    const logic: loop.Logic(Args, Args) = struct {
        fn logic(b1: *const VB, b2: *VB, idx: []const usize) void {
            const val = b1.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    nest.eval(Args, Args, 2, .{vb}, .{vb}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** 128), b.data[0..128]);
}
