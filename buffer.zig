const std = @import("std");
const loop = @import("loop.zig");

fn ShapeToArray(comptime dtype: type, comptime ndims: u8, comptime shape: *const [ndims]usize) type {
    var Type = dtype;
    for (0..shape.len) |dim| {
        const s = shape[ndims - dim - 1];
        Type = [s]Type;
    }
    return Type;
}

fn Datatype(comptime Array: type) type {
    switch (@typeInfo(Array)) {
        .Vector => |info| return info.child,
        .Pointer => |info| if (info.size == .Many) return IterationSpace(info.child).dtype,
        .Array => |info| return IterationSpace(info.child).dtype,
        .Int, .Float, .Bool => return Array,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

fn extractNdims(comptime Array: type) u8 {
    switch (@typeInfo(Array)) {
        .Pointer => |info| if (info.size == .Many) return 1 + IterationSpace(info.child).ndims,
        .Array => |info| return 1 + IterationSpace(info.child).ndims,
        .Int, .Float, .Bool => return 0,
        .Vector => return 1,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

fn extractShape(comptime Array: type) [extractNdims(Array)]usize {
    switch (@typeInfo(Array)) {
        .Pointer => |info| if (info.size == .Many) return .{null} ++ extractShape(info.child),
        .Array => |info| return .{info.len} ++ extractShape(info.child),
        .Int, .Float, .Bool => return .{},
        .Vector => |info| {
            std.debug.assert(info.child == Datatype(Array));
            return .{info.len};
        },
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

fn Unit(comptime Array: type) type {
    const dtype = Datatype(Array);
    const ndims = extractNdims(Array);
    const shape: [ndims]usize = extractShape(Array);

    switch (@typeInfo(Array)) {
        .Pointer => |info| if (info.size == .Many) return Unit(info.child),
        .Array => |info| return Unit(info.child),
        .Int, .Float, .Bool => return dtype,
        .Vector => return @Vector(shape[ndims - 1], dtype),
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

pub fn Vectorized(comptime Array: type) type {
    const dtype = Datatype(Array);
    const ndims = extractNdims(Array);
    const shape = extractShape(Array);
    const Vec: type = @Vector(shape[ndims - 1], dtype);

    // do not vectorize a buffer that is already vectorized.
    if (Unit(Array) == Datatype(Array)) {
        var Type: type = Vec;
        for (0..ndims - 1) |dim| {
            Type = [shape[ndims - dim - 2]]Type;
        }
        return Type;
    }
    @compileError("cannot vectorize a buffer that is already vectorized");
}

pub const Layout = struct {
    ndims: u8,
    shape: []const usize,
    strides: []const usize,
    offest: usize = 0,
};

pub fn AllocatedBuffer(comptime Array: type) type {
    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;
        const SIMD_ALIGN = @alignOf(Vec);
        const alignment = @max(SIMD_ALIGN, CACHE_LINE);

        layout: Layout,
        multi: *Array align(alignment),
        raw: [*]dtype align(alignment),

        const dtype = Datatype(Array);
        const ndims = extractNdims(Array);
        const shape = extractShape(Array);
        /// Vectorized contiguous dim of the buffer.
        pub const Vec: type = @Vector(shape[ndims - 1], dtype);
        pub const Arr: type = Array;

        const ThisUnit = Unit(Array);
        const ThisVectorized = Vectorized(Array);

        const numel: ?comptime_int = blk: {
            var n = 1;
            for (shape) |len| {
                n *= len;
            }
            break :blk n;
        };

        const contiguous_strides: [ndims]usize = blk: {
            if (ndims == 0) {
                break :blk .{};
            }
            var offset: usize = 1;
            var strides: [ndims]usize = undefined;
            for (0..ndims - 1) |d| {
                const stride = shape[ndims - d - 1] * offset;
                strides[ndims - d - 2] = stride;
                offset = stride;
            }
            strides[ndims - 1] = 1;
            for (0..ndims) |d| {
                if (shape[d] == 0 or shape[d] == 1) {
                    strides[d] = 0;
                }
            }
            break :blk strides;
        };

        pub fn alloc(allocator: std.mem.Allocator) !AllocatedBuffer(Arr) {
            const slice: []align(alignment) dtype = try allocator.alignedAlloc(dtype, alignment, numel.?);
            if (dtype == bool) {
                @memset(slice, false);
            } else {
                @memset(slice, 0);
            }

            return AllocatedBuffer(Arr){
                .layout = .{
                    .ndims = ndims,
                    .shape = &shape,
                    .strides = &contiguous_strides,
                    // .skew = &(.{0} ** ndims),
                    // .unrolled = &(.{false} ** ndims),
                    // .parallelized = &(.{false} ** ndims),
                },
                .multi = @alignCast(@ptrCast(slice)),
                .raw = @alignCast(@ptrCast(slice)),
            };
        }

        pub fn constant(_: Self, val: dtype) ThisUnit {
            if (ThisUnit == dtype) {
                return val;
            } else {
                return @splat(val);
            }
        }

        pub inline fn unravel(b: Self, idx: [ndims]usize) usize {
            var i: usize = 0;
            inline for (0..ndims) |d| {
                comptime if (ThisUnit != dtype and d == ndims - 1) break;
                i += idx[d] * b.layout.strides[d];
            }
            return i;
        }

        pub inline fn load(b: Self, idx: [ndims]usize) ThisUnit {
            if (comptime ThisUnit != dtype) {
                const val: *const ThisUnit align(alignment) = @alignCast(@ptrCast(b.raw[b.unravel(idx)..][0..shape[ndims - 1]]));
                return @as(ThisUnit, val.*);
            } else {
                return b.raw[b.unravel(idx)];
            }
        }

        pub inline fn store(b: Self, val: ThisUnit, idx: [ndims]usize) void {
            if (comptime ThisUnit != dtype) {
                const dst: [*]ThisUnit align(alignment) = @alignCast(@ptrCast(b.raw[b.unravel(idx)..][0..shape[ndims - 1]]));
                // TODO: Need to verify what asm this generates, needs to be vmovaps
                @memcpy(@as([*]dtype, @ptrCast(dst)), &@as([shape[ndims - 1]]dtype, val));
            } else {
                b.raw[b.unravel(idx)] = val;
            }
        }
    };
}

pub const BlockInfo = struct {
    orig_dim: u8,
    num_blocks: usize,
    block_size: usize,
};

pub fn IterationSpace(comptime Array: type) type {
    return struct {
        const Self = @This();
        const dtype: type = Datatype(Array);
        const ndims = extractNdims(Array);
        const shape: [ndims]usize = extractShape(Array);
        const ThisUnit = Unit(Array);
        const ThisVectorized = Vectorized(Array);
        const default_block_info = blk: {
            var splits: [ndims]BlockInfo = undefined;
            for (0..ndims) |d| {
                splits[d] = .{
                    .orig_dim = d,
                    .block_size = 1,
                    .num_blocks = shape[d],
                };
            }
            break :blk splits;
        };

        orig_ndims: u8,
        block_info: *const [ndims]BlockInfo = &default_block_info,
        vector: bool = (ThisUnit == Vec),

        pub fn init() Self {
            return .{
                .orig_ndims = ndims,
            };
        }

        fn Split(comptime dim: u8, comptime block_size: u8) type {
            std.debug.assert(dim < ndims);
            // TODO: Support splits that don't divide evenly
            // Give the option of how to evaluate the uneven part
            // - Pad it to evenly divide
            // - Unfuse into two separate loops (gives more control for unrolling)
            std.debug.assert(@mod(shape[dim], block_size) == 0);

            if (block_size == 1 or block_size == shape[dim]) {
                return Self;
            }

            const num_blocks = @divExact(shape[dim], block_size);

            const pre = if (dim > 0) shape[0..dim] else .{};
            const post = if (dim < ndims - 1) shape[dim + 1 .. ndims] else .{};
            return IterationSpace(ShapeToArray(dtype, ndims + 1, pre ++ .{ num_blocks, block_size } ++ post));
        }

        pub fn split(comptime b: *const Self, comptime dim: u8, comptime block_size: u8) Split(dim, block_size) {
            if (Split(dim, block_size) == Self) {
                return b.*;
            }

            const block_info1: BlockInfo = .{
                .orig_dim = b.block_info[dim].orig_dim,
                .num_blocks = Split(dim, block_size).shape[dim],
                .block_size = block_size,
            };

            const block_info2: BlockInfo = .{
                .orig_dim = b.block_info[dim].orig_dim,
                .num_blocks = Split(dim, block_size).shape[dim + 1],
                .block_size = b.block_info[dim].block_size,
            };

            return .{
                .orig_ndims = b.orig_ndims,
                .block_info = b.block_info[0..dim] ++ .{ block_info1, block_info2 } ++ b.block_info[dim + 1 .. ndims],
            };
        }

        pub const Vec: type = @Vector(shape[ndims - 1], dtype);
        pub const Arr = Array;

        // By setting NonVectorized to void if it is already vectorized
        // the following function is removed from the namespace of this type
        const NonVectorized = if (ThisVectorized != void) Self else void;
        pub fn vectorize(b: *const NonVectorized) IterationSpace(ThisVectorized) {
            return .{ .block_info = b.block_info, .orig_ndims = b.orig_ndims };
        }

        fn buildLoop(comptime b: *const Self, comptime dim: u8, inner: ?*const loop.Nest.Loop) ?loop.Nest.Loop {
            if (dim >= ndims) {
                return null;
            }

            return .{
                // .lower = b.layout.skew[dim],
                .lower = 0,
                .upper = shape[dim],
                .inner = inner,
                .block_info = b.block_info[dim],
                .vector = (ThisUnit == Vec and dim == ndims - 1),
                .step_size = if (ThisUnit == Vec and dim == ndims - 1) shape[ndims - 1] else 1,
            };
        }

        pub fn nest(
            comptime b: *const Self,
            comptime float_mode: std.builtin.FloatMode,
        ) loop.Nest {
            if (ndims == 0) {
                @compileError("cannot generate loop nest for 0 dimensional iteration space");
            }

            const loop_nest: ?loop.Nest.Loop = comptime blk: {
                var curr: ?loop.Nest.Loop = null;
                for (0..ndims) |dim| {
                    if (curr) |curr_loop| {
                        curr = b.buildLoop(ndims - dim - 1, &curr_loop);
                    } else {
                        curr = b.buildLoop(ndims - dim - 1, null);
                    }
                }
                break :blk curr;
            };

            if (loop_nest) |top_loop| {
                return .{
                    .orig_ndims = b.orig_ndims,
                    .float_mode = float_mode,
                    .loop = &top_loop,
                };
            } else {
                return .{
                    .orig_ndims = b.orig_ndims,
                    .float_mode = float_mode,
                    .loop = null,
                };
            }
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const t = comptime IterationSpace([16][8]f32).init();
    var d = try AllocatedBuffer(@TypeOf(t).Arr).alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&d.multi[0]) == @intFromPtr(&d.raw[0]));
    try std.testing.expect(@intFromPtr(&d.multi[1][7]) == @intFromPtr(&d.raw[15]));
}

test "split" {
    const st = comptime IterationSpace([16][8]f32).init().split(0, 4);
    try std.testing.expect(@TypeOf(st) == IterationSpace([4][4][8]f32));
    try std.testing.expectEqualSlices(BlockInfo, &.{
        .{
            .orig_dim = 0,
            .block_size = 4,
            .num_blocks = 4,
        },
        .{
            .orig_dim = 0,
            .block_size = 1,
            .num_blocks = 4,
        },
        .{
            .orig_dim = 1,
            .block_size = 1,
            .num_blocks = 8,
        },
    }, st.block_info);
}

test "split_split" {
    const sst = comptime IterationSpace([16][8]f32).init().split(0, 4).split(0, 2);
    try std.testing.expect(@TypeOf(sst) == IterationSpace([2][2][4][8]f32));
    try std.testing.expectEqualSlices(BlockInfo, &.{ .{
        .orig_dim = 0,
        .block_size = 2,
        .num_blocks = 2,
    }, .{
        .orig_dim = 0,
        .block_size = 4,
        .num_blocks = 2,
    }, .{
        .orig_dim = 0,
        .block_size = 1,
        .num_blocks = 4,
    }, .{
        .orig_dim = 1,
        .num_blocks = 8,
        .block_size = 1,
    } }, sst.block_info);
}

test "vectorize" {
    const t = comptime IterationSpace([16][8]f32).init();
    const vt = t.vectorize();
    try std.testing.expect(@TypeOf(vt) == IterationSpace([16]@Vector(8, f32)));
}

test "split_vectorize" {
    const svt = comptime IterationSpace([16][8]f32).init()
        .split(0, 4)
        .vectorize();
    try std.testing.expect(@TypeOf(svt) == IterationSpace([4][4]@Vector(8, f32)));
    try std.testing.expectEqualSlices(BlockInfo, &.{ .{
        .orig_dim = 0,
        .block_size = 4,
        .num_blocks = 4,
    }, .{
        .orig_dim = 0,
        .block_size = 1,
        .num_blocks = 4,
    }, .{
        .orig_dim = 1,
        .num_blocks = 8,
        .block_size = 1,
    } }, svt.block_info);
}

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const t = comptime IterationSpace([16][8]f32).init();
    const nest = comptime t.nest(.optimized);
    const expected = comptime &loop.Nest.Loop{
        .upper = 16,
        .block_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .block_size = 1,
        },
        .inner = &.{
            .upper = 8,
            .inner = null,
            .block_info = .{
                .num_blocks = 8,
                .orig_dim = 1,
                .block_size = 1,
            },
        },
    };
    try std.testing.expectEqualDeep(expected, nest.loop);
}

test "vectorize_nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const t = comptime IterationSpace([16][8]f32).init().vectorize();
    defer arena.deinit();

    const nest = comptime t.nest(.optimized);
    const expected = comptime &loop.Nest.Loop{
        .upper = 16,
        .block_info = .{
            .block_size = 1,
            .num_blocks = 16,
            .orig_dim = 0,
        },
        .inner = &loop.Nest.Loop{
            .upper = 8,
            .vector = true,
            .step_size = 8,
            .block_info = .{
                .num_blocks = 8,
                .block_size = 1,
                .orig_dim = 1,
            },
        },
    };
    try std.testing.expectEqualDeep(expected, nest.loop);
}

test "nest_eval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const t = comptime IterationSpace([16][8]i32).init();
    defer arena.deinit();

    const nest = comptime t.nest(.optimized);
    const B = @TypeOf(t).Arr;
    const Args = struct {
        b: B,
    };

    const logic: loop.Logic(Args, Args, AllocatedBuffer(B).ndims) = struct {
        inline fn logic(b1: *const AllocatedBuffer(B), b2: *AllocatedBuffer(B), idx: [AllocatedBuffer(B).ndims]usize) void {
            const val = b2.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    nest.eval(Args, Args, .{&b}, .{&b}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** 128), b.raw[0..128]);
}

test "vectorize_nest_eval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const t = comptime IterationSpace([16][8]i32).init().vectorize();
    try std.testing.expect(@TypeOf(t).Vec == @Vector(8, i32));
    const B = [16][8]i32;
    const Args = struct {
        b: B,
    };

    const logic: loop.Logic(Args, Args, AllocatedBuffer(B).ndims) = comptime struct {
        // Idea: using callconv() require a GPU kernel here, or inline this function
        // into a surrounding GPU kernel. Unraveling method would require to be bound to GPU
        // thread / group ids
        inline fn logic(b1: *const AllocatedBuffer(B), b2: *AllocatedBuffer(B), idx: [AllocatedBuffer(B).ndims]usize) void {
            const val = b2.load(idx) + b1.constant(1);
            b2.store(val, idx);
        }
    }.logic;

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    const nest = comptime t.nest(.optimized);
    nest.eval(Args, Args, .{&b}, .{&b}, logic);
    try std.testing.expectEqualSlices(i32, &(.{1} ** 128), b.raw[0..128]);
}

test "double_split_vectorized_nest_eval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const t = comptime IterationSpace([128][4]bool).init()
        .split(0, 16)
        .split(1, 4)
        .vectorize();
    try std.testing.expect(@TypeOf(t) == IterationSpace([8][4][4]@Vector(4, bool)));

    const B = [128][4]bool;
    const Args = struct { b: B };
    const logic: loop.Logic(void, Args, AllocatedBuffer(B).ndims) = struct {
        inline fn logic(b: *AllocatedBuffer(B), idx: [AllocatedBuffer(B).ndims]usize) void {
            b.store(b.constant(true), idx);
        }
    }.logic;

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    const nest = comptime t.nest(.optimized);
    const expected = comptime &loop.Nest.Loop{
        .upper = 8,
        .block_info = .{
            .num_blocks = 8,
            .block_size = 16,
            .orig_dim = 0,
        },
        .inner = &loop.Nest.Loop{
            .upper = 4,
            .block_info = .{
                .num_blocks = 4,
                .block_size = 4,
                .orig_dim = 0,
            },
            .inner = &loop.Nest.Loop{
                .upper = 4,
                .block_info = .{
                    .num_blocks = 4,
                    .block_size = 1,
                    .orig_dim = 0,
                },
                .inner = &loop.Nest.Loop{
                    .upper = 4,
                    .step_size = 4,
                    .vector = true,
                    .block_info = .{
                        .num_blocks = 4,
                        .block_size = 1,
                        .orig_dim = 1,
                    },
                },
            },
        },
    };
    nest.eval(void, Args, .{}, .{&b}, logic);
    try std.testing.expectEqualDeep(expected, nest.loop);
    try std.testing.expectEqualSlices(bool, &(.{true} ** 512), b.raw[0..512]);
}
