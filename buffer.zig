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
    ndims: u8,
    shape: []const usize,
    strides: []const usize,
    offest: usize = 0,
};

pub fn Tensor(comptime Array: type) type {
    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;
        const SIMD_ALIGN = @alignOf(Vec);

        const alignment = @max(SIMD_ALIGN, CACHE_LINE);

        pub const Buffer = struct {
            layout: Layout,
            multi: *Array align(alignment),
            raw: [*]dtype align(alignment),

            pub fn constant(_: Buffer, val: dtype) Base {
                if (Base == dtype) {
                    return val;
                } else {
                    return @splat(val);
                }
            }

            pub inline fn unravel(b: Buffer, idx: []const usize) usize {
                var i: usize = 0;
                inline for (0..t_ndims) |d| {
                    comptime if (Base != dtype and d == t_ndims - 1) break;
                    i += idx[d] * b.layout.strides[d];
                }
                return i;
            }

            pub inline fn load(b: Buffer, idx: []const usize) Base {
                if (comptime Base != dtype) {
                    const val: *const Base align(alignment) = @alignCast(b.raw[b.unravel(idx)..][0..t_shape[t_ndims - 1]]);
                    return @as(Base, val.*);
                } else {
                    return b.raw[b.unravel(idx)];
                }
            }

            pub inline fn store(b: Buffer, val: Base, idx: []const usize) void {
                if (comptime Base != dtype) {
                    const dst: *Base align(alignment) = @alignCast(b.raw[b.unravel(idx)..][0..t_shape[t_ndims - 1]]);
                    dst.* = val;
                } else {
                    b.raw[b.unravel(idx)] = val;
                }
            }
        };

        layout: Layout,
        pub fn init() Self {
            return comptime .{
                .layout = .{
                    .ndims = t_ndims,
                    .shape = &t_shape,
                    .strides = &t_contiguous_strides,
                    // .skew = &(.{0} ** t_ndims),
                    // .unrolled = &(.{false} ** t_ndims),
                    // .parallelized = &(.{false} ** t_ndims),
                },
            };
        }

        pub fn alloc(self: *const Self, allocator: std.mem.Allocator) !Buffer {
            const slice: []align(alignment) dtype = try allocator.alignedAlloc(dtype, alignment, t_numel.?);
            if (dtype == bool) {
                @memset(slice, false);
            } else {
                @memset(slice, 0);
            }

            return Buffer{
                .layout = self.layout,
                .multi = @alignCast(@ptrCast(slice)),
                .raw = @alignCast(@ptrCast(slice)),
            };
        }

        pub const dtype: type = blk: {
            switch (@typeInfo(Array)) {
                .Vector => |info| break :blk info.child,
                .Pointer => |info| if (info.size == .Many) break :blk Tensor(info.child).dtype,
                .Array => |info| break :blk Tensor(info.child).dtype,
                .Int, .Float, .Bool => break :blk Array,
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        const t_ndims = blk: {
            switch (@typeInfo(Array)) {
                .Pointer => |info| if (info.size == .Many) break :blk 1 + Tensor(info.child).t_ndims,
                .Array => |info| break :blk 1 + Tensor(info.child).t_ndims,
                .Int, .Float, .Bool => break :blk 0,
                .Vector => break :blk 1,
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        const t_shape: [t_ndims]usize = blk: {
            switch (@typeInfo(Array)) {
                .Pointer => |info| if (info.size == .Many) break :blk .{null} ++ Tensor(info.child).t_shape,
                .Array => |info| break :blk .{info.len} ++ Tensor(info.child).t_shape,
                .Int, .Float, .Bool => break :blk .{},
                .Vector => |info| {
                    std.debug.assert(Tensor(info.child).dtype == dtype);
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
                .Pointer => |info| if (info.size == .Many) break :blk Tensor(info.child).Base,
                .Array => |info| break :blk Tensor(info.child).Base,
                .Int, .Float, .Bool => break :blk dtype,
                .Vector => break :blk @Vector(t_shape[t_ndims - 1], dtype),
                else => {},
            }
            @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
        };

        fn Split(comptime dim: u8, comptime count: u8) type {
            std.debug.assert(dim < t_ndims);
            // TODO: Support splits that don't divide evenly
            // Give the option of how to evaluate the uneven part
            // - Pad it to evenly divide
            // - Unfuse into two separate loops (gives more control for unrolling)
            std.debug.assert(@mod(t_shape[dim], count) == 0);

            if (count == 1 or count == t_shape[dim]) {
                return Self;
            }

            const pre = if (dim > 0) t_shape[0..dim].* else .{};
            const post = if (dim < t_ndims - 1) t_shape[dim + 1 .. t_ndims].* else .{};
            return Tensor(ShapeToArray(dtype, t_ndims + 1, pre ++ .{ @divExact(t_shape[dim], count), count } ++ post));
        }

        pub fn split(comptime b: *const Self, comptime dim: u8, comptime count: u8) Split(dim, count) {
            if (Split(dim, count) == Self) {
                return b.*;
            }

            var layout: Layout = b.layout;
            layout.ndims += 1;
            layout.shape = &Split(dim, count).t_shape;

            const pre: [dim]usize = if (dim > 0) b.layout.strides[0..dim].* else .{};
            const new: [2]usize = .{ b.layout.strides[dim] * count, b.layout.strides[dim] };
            const post: [t_ndims - dim - 1]usize = if (dim < t_ndims - 1) b.layout.strides[dim + 1 .. t_ndims].* else .{};
            layout.strides = &(pre ++ new ++ post);

            return .{
                .layout = layout,
            };
        }

        const Vec: type = @Vector(t_shape[t_ndims - 1], dtype);
        /// Vectorized contiguous dim of the buffer.
        pub const Vectorized: type = t: {
            // do not vectorize a buffer that is already vectorized.
            if (Base == dtype) {
                var Type: type = Vec;
                for (0..t_ndims - 1) |dim| {
                    Type = [t_shape[t_ndims - dim - 2]]Type;
                }
                break :t Tensor(Type);
            }

            break :t void;
        };

        // By setting NonVectorized to void if it is already vectorized
        // the following function is removed from the namespace of this type
        const NonVectorized = if (Vectorized != void) Self else void;
        pub fn vectorize(b: *const NonVectorized) Vectorized {
            std.debug.assert(b.layout.strides[b.layout.ndims - 1] == 1);
            return .{ .layout = b.layout };
        }

        fn buildLoop(comptime b: *const Self, comptime dim: u8, inner: ?*const loop.Loop) ?loop.Loop {
            if (dim >= t_ndims) {
                return null;
            }
            if (Base == Vec and dim == b.layout.ndims - 1) {
                return null;
            } else {
                return .{
                    // .lower = b.layout.skew[dim],
                    .lower = 0,
                    .upper = b.layout.shape[dim],
                    .inner = inner,
                };
            }
        }

        pub fn nest(
            comptime b: *const Self,
            comptime float_mode: std.builtin.FloatMode,
        ) loop.Nest {
            if (b.layout.ndims == 0) {
                @compileError("cannot generate loop nest for 0 dimensional buffer");
            }

            const loop_nest: ?loop.Loop = comptime blk: {
                var curr: ?loop.Loop = null;
                for (0..b.layout.ndims) |dim| {
                    if (curr) |curr_loop| {
                        curr = b.buildLoop(b.layout.ndims - dim - 1, &curr_loop);
                    } else {
                        curr = b.buildLoop(b.layout.ndims - dim - 1, null);
                    }
                }
                break :blk curr;
            };

            if (loop_nest) |top_loop| {
                return .{
                    .ndims = b.layout.ndims,
                    .float_mode = float_mode,
                    .loop = &top_loop,
                };
            } else {
                return .{
                    .ndims = b.layout.ndims,
                    .float_mode = float_mode,
                    .loop = null,
                };
            }
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const t = comptime Tensor([16][8]f32).init();
    var d = try t.alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&d.multi[0]) == @intFromPtr(&d.raw[0]));
    try std.testing.expect(@intFromPtr(&d.multi[1][7]) == @intFromPtr(&d.raw[15]));
}

test "split" {
    const t = comptime Tensor([16][8]f32).init().split(0, 4);
    try std.testing.expect(@TypeOf(t) == Tensor([4][4][8]f32));
    try std.testing.expectEqualSlices(usize, &.{ 32, 8, 1 }, t.layout.strides);
}

test "vectorize" {
    const t = comptime Tensor([16][8]f32).init();
    const vt = t.vectorize();
    try std.testing.expect(@TypeOf(vt) == Tensor([16]@Vector(8, f32)));
}

test "split_vectorize" {
    const t = comptime Tensor([16][8]f32).init();
    const svt = comptime t
        .split(0, 4)
        .vectorize();
    try std.testing.expect(@TypeOf(svt) == Tensor([4][4]@Vector(8, f32)));
    try std.testing.expectEqualSlices(usize, &.{ 32, 8, 1 }, svt.layout.strides);
}

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const expected = comptime &loop.Loop{
        .upper = 16,
        .inner = &.{
            .upper = 8,
            .inner = null,
        },
    };
    const t = comptime Tensor([16][8]f32).init();
    const nest = comptime t.nest(.optimized);
    try std.testing.expectEqualDeep(expected, nest.loop);
}

test "vectorize_nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const expected = comptime &loop.Loop{
        .upper = 16,
        .inner = null,
    };
    const t = comptime Tensor([16][8]f32).init().vectorize();
    defer arena.deinit();

    const nest = comptime t.nest(.optimized);
    try std.testing.expectEqualDeep(expected, nest.loop);
}

test "nest_eval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const t = comptime Tensor([16][8]i32).init();
    defer arena.deinit();

    const nest = comptime t.nest(.optimized);
    const B = comptime Tensor([16][8]i32).Buffer;
    const Args = struct {
        b: B,
    };

    const logic: loop.Logic(Args, Args) = struct {
        inline fn logic(b1: *const B, b2: *B, idx: []const usize) void {
            const val = b2.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    var b = try t.alloc(arena.allocator());
    nest.eval(Args, Args, .{&b}, .{&b}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** 128), b.raw[0..128]);
}

test "vectorize_nest_eval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const t = comptime Tensor([16][8]i32).init().vectorize();
    try std.testing.expect(@TypeOf(t).Base == @Vector(8, i32));
    const B = @TypeOf(t).Buffer;
    const Args = struct {
        b: B,
    };

    const logic: loop.Logic(Args, Args) = comptime struct {
        // Idea: using callconv() require a GPU kernel here, or inline this function
        // into a surrounding GPU kernel. Unraveling method would require to be bound to GPU
        // thread / group ids
        inline fn logic(b1: *const B, b2: *B, idx: []const usize) void {
            const val = b2.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    var b = try t.alloc(arena.allocator());
    const nest = comptime t.nest(.optimized);
    nest.eval(Args, Args, .{&b}, .{&b}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** 128), b.raw[0..128]);
}

test "double_split_vectorize_nest_eval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const t = comptime Tensor([128][4]i32).init()
        .split(0, 16)
        .split(1, 4)
        .vectorize();

    const B = @TypeOf(t).Buffer;
    try std.testing.expect(@TypeOf(t) == Tensor([8][4][4]@Vector(4, i32)));

    const Args = struct { b: B };
    const logic: loop.Logic(Args, Args) = struct {
        inline fn logic(b1: *const B, b2: *B, idx: []const usize) void {
            const val = b2.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    var b = try t.alloc(arena.allocator());

    const nest = comptime t.nest(.optimized);
    nest.eval(Args, Args, .{&b}, .{&b}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** (128 * 4)), b.raw[0..(128 * 4)]);
}
