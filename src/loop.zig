const std = @import("std");
const utils = @import("utils.zig");
const func = @import("func.zig");

const buffer = @import("buffer.zig");
const IterSpace = @import("IterSpace.zig");

pub fn Nest(comptime Args: type, comptime iter_space: IterSpace) type {
    return struct {
        const idx_ndims = iter_space.numDataIndices();
        const CurrentNest = @This();
        const EvalItem = struct {
            eval: func.IterSpaceFunc(Args, iter_space).TupleFn,
        };

        body: []const EvalItem,

        pub fn init(
            comptime iter_logic: func.Func(Args, iter_space.DataIndex).Def,
        ) @This() {
            if (iter_space.ndims() == 0) {
                @compileError("cannot generate loop nest for 0 dimensional iteration space");
            }
            return .{
                .body = comptime blk: {
                    var body: []const EvalItem = &.{buildFunc(iter_logic)};
                    for (0..iter_space.ndims()) |dim| {
                        body = &.{buildLoop(iter_space.ndims() - dim - 1, body)};
                    }
                    break :blk &.{body[0]};
                },
            };
        }

        fn buildKernel(comptime iter_logic: func.Func(Args, iter_space.DataIndex).Def) EvalItem {
            return .{
                .eval = struct {
                    const DeviceKernel = func.IterSpaceKernel(Args, iter_space);
                    // The kernel func will look exactly like the iter logic, except it needs to have
                    // the iter space specified and also have the Kernel calling convention
                    const kernel = @as(*const DeviceKernel, @ptrCast(&iter_logic));
                    // The eval function launches the kernel
                    inline fn eval(idx: [idx_ndims]usize, args: func.IterSpaceArgsTuple(Args, iter_space)) void {
                        // TODO: copy args to gpu device
                        const tuple_args: *const func.IterSpaceArgsTuple(Args, iter_space) = @ptrCast(&args);
                        @call(.auto, kernel, .{idx} ++ tuple_args.*);
                    }
                }.eval,
            };
        }

        fn buildFunc(comptime iter_logic: func.Func(Args, iter_space.DataIndex).Def) EvalItem {
            return .{
                .eval = struct {
                    // This
                    const EvalFunc: type = func.IterSpaceFunc(Args, iter_space).SeparatesFn;
                    const evalFunc = @as(*const EvalFunc, @ptrCast(&iter_logic));
                    /// Buffers are initially created with a default iter_space in the type, corresponding to the array type,
                    /// not the one that has been transformed by the programmer.
                    /// Both buffers are identical but the additional type information from the iter_space being used will change
                    /// Casting will allow the usage of the data indices enum to correctly identify vector loads and stores
                    inline fn eval(idx: [idx_ndims]usize, args: func.IterSpaceArgsTuple(Args, iter_space)) void {
                        const tuple_args: *const func.IterSpaceArgsTuple(Args, iter_space) = @ptrCast(&args);
                        @call(.always_inline, evalFunc, .{idx} ++ tuple_args.*);
                    }
                }.eval,
            };
        }

        fn buildLoop(comptime dim: u8, body: []const EvalItem) EvalItem {
            return .{
                .eval = struct {
                    const loop_info = iter_space.loop_info[dim];
                    /// Heart of the runtime code, this function will use comptime-powered elision to
                    /// only generate / execute sections that are relevant to the loop
                    /// - If the loop is not unrolled, only the regular loop branch will generate code
                    /// - If the loop is not parallelized, no code for thead spawning/joining will be generated
                    ///
                    /// Similarly, the buffers will check during compile time whether to do a vector or scalar load
                    inline fn eval(base_idx: [idx_ndims]usize, args: func.IterSpaceArgsTuple(Args, iter_space)) void {
                        var idx = base_idx;
                        const base_for_dim = base_idx[loop_info.idx_dim];
                        // if the loop is not parallel this should be a noop
                        var threads: [loop_info.parallel orelse 0]std.Thread = undefined;
                        inline for (body) |item| {
                            if (comptime loop_info.unrolled) {
                                // for inlining, the loopvar is comptime
                                comptime var unroll_loopvar = loop_info.idx_min;
                                comptime var i = 0;
                                // null assert required as loop bound must be comptime known (cannot be dynamic)
                                inline while (i < loop_info.num_blocks.?) : ({
                                    unroll_loopvar += loop_info.block_size;
                                    idx[loop_info.idx_dim] = base_for_dim + unroll_loopvar;
                                    i += 1;
                                }) {
                                    if (comptime loop_info.parallel) |_| {
                                        threads[i] = std.Thread.spawn(.{}, item.eval, .{ idx, args }) catch unreachable;
                                    } else {
                                        item.eval(idx, args);
                                    }
                                }
                            } else {
                                var loopvar = loop_info.idx_min;
                                var i: usize = 0;
                                while (i < loop_info.num_blocks.?) : ({
                                    loopvar += loop_info.block_size;
                                    idx[loop_info.idx_dim] = base_for_dim + loopvar;
                                    i += 1;
                                }) {
                                    if (comptime loop_info.parallel) |_| {
                                        threads[i] = std.Thread.spawn(.{}, item.eval, .{ idx, args }) catch unreachable;
                                    } else {
                                        item.eval(idx, args);
                                    }
                                }
                            }
                            // this would also be a noop if there are 0 parallel threads
                            inline for (threads) |t| {
                                t.join();
                            }
                        }
                    }
                }.eval,
            };
        }

        pub fn compile(comptime nest: *const @This()) func.ExternFunc(Args, iter_space).Def {
            return comptime struct {
                fn eval(args: func.ExternFunc(Args, iter_space).Param) callconv(.C) void {
                    const idx: [idx_ndims]usize = .{0} ** idx_ndims;
                    var _args: func.IterSpaceArgsTuple(Args, iter_space) = undefined;
                    inline for (comptime std.meta.fieldNames(Args), 0..) |field_name, field_idx| {
                        @field(_args, std.fmt.comptimePrint("{d}", .{field_idx})) = @field(args, field_name);
                    }
                    inline for (nest.body) |item| {
                        item.eval(idx, _args);
                    }
                }
            }.eval;
        }

        pub inline fn eval(
            comptime nest: *const @This(),
            args: func.ExternFuncParam(Args, iter_space),
        ) void {
            const eval_func = comptime nest.compile();
            eval_func(args);
        }

        pub fn alloc(
            comptime _: *const @This(),
            comptime Array: type,
            allocator: std.mem.Allocator,
        ) !buffer.IterSpaceBuffer(Array, iter_space) {
            return buffer.IterSpaceBuffer(Array, iter_space).alloc(allocator);
        }
        // fuse will need to fuse the args of two different nests
        // to build a new type and appropriately pass in the args into the functions
        // which require the originally defined arg types, this will be tricky!
        // alternatively, have an execution wrapper that manages arg passing and executes loops / logic manually, and keep nests separate
    };
}

const TestIndices = enum { i, j };

const test_iter_space = IterSpace.init([16][8]f32, TestIndices);
const B = [16][8]bool;
const TestArgs = struct {
    b: B,
};

const Func = func.Func(TestArgs, TestIndices);
const test_logic: Func.Def = struct {
    // for gpu execution inline this function into a surrounding GPU kernel.
    // Unraveling method would require to be bound to GPU thread / group ids
    inline fn iter_logic(idx: [2]usize, b: Func.Param(B)) void {
        std.testing.expectEqual(b.constant(.{ .i, .j }, false), b.load(.{ .i, .j }, idx)) catch unreachable;
        b.store(.{ .i, .j }, b.constant(.{ .i, .j }, true), idx);
    }
}.iter_logic;

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const nest = test_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .num_blocks = 16,
        .idx_dim = 0,
    }, .{
        .num_blocks = 8,
        .idx_dim = 1,
    } }, test_iter_space.loop_info);
    var b = try nest.alloc(B, arena.allocator());

    nest.eval(.{ .b = b });
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "runtime bounds" {
    // TODO: To support runtime array sizes, the eval function will need to have extra args for runtime sizes.
    // This should still be doable, can maybe construct a new type from Args type that identifies
    // runtime bounds given by [*] and converts them to a tuple type where the bound must be specified
    // e.g. [16][*]f32 => std.meta.Tuple(&.{?usize, usize})
    // Args = struct { a: [16][*]f32 } => RuntimeSizes = struct { a: std.meta.Tuple(&.{?usize, usize}) }
    // Obviously, unrolling, vectorization won't be allowed.
    // Splitting might have some restrictions applied and will definitely need to have boundary handling
    // Parallelism might also get a little weird, we can technically launch as many threads as we want but
    // there would be runtime overhead to allocate the array for threads (either arraylist or alloc)
    // GPU would be tbd

    // var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    // defer arena.deinit();
    // const nest = test_iter_space.nest(TestArgs, test_logic);
    // try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
    //     .num_blocks = 16,
    //     .idx_dim = 0,
    //     .vector = false,
    //     .block_size = 1,
    //     .unrolled = false,
    //     .parallel = false,
    // }, .{
    //     .num_blocks = 8,
    //     .idx_dim = 1,
    //     .vector = false,
    //     .block_size = 1,
    //     .unrolled = false,
    //     .parallel = false,
    // } }, test_iter_space.loop_info);

    // var b = try nest.alloc(B, arena.allocator());

    // nest.eval(.{.b = b});
    // try std.testing.expectEqualSlices(bool, &(.{true} ** 64 ++ .{false} ** 64), b.data[0..128]);
}

test "unroll nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const unrolled_iter_space = comptime test_iter_space.unroll(.j);
    const nest = unrolled_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{
        .{
            .num_blocks = 16,
            .idx_dim = 0,
        },
        .{
            .num_blocks = 8,
            .idx_dim = 1,
            .unrolled = true,
        },
    }, unrolled_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{ .b = b });
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const vectorized_iter_space = comptime test_iter_space.vectorize(.j);
    const nest = comptime vectorized_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .num_blocks = 16,
        .idx_dim = 0,
    }, .{
        .num_blocks = 1,
        .idx_dim = 1,
        .block_size = 8,
        .vector = true,
    } }, vectorized_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{ .b = b });
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "reorder nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const reordered_iter_space = comptime test_iter_space.reorder(.{ 1, 0 });
    const nest = comptime reordered_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .num_blocks = 8,
        .idx_dim = 1,
    }, .{
        .num_blocks = 16,
        .idx_dim = 0,
    } }, reordered_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{ .b = b });
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "parallel nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const parallel_iter_space = comptime test_iter_space.parallel(.j, null);
    const nest = comptime parallel_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .block_size = 1,
        .num_blocks = 16,
        .idx_dim = 0,
    }, .{
        .num_blocks = 8,
        .idx_dim = 1,
        .block_size = 1,
        .parallel = 8,
    } }, parallel_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{ .b = b });
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "split split vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const transformed_iter_space = comptime test_iter_space
        .split(.j, 4)
        .split(.i, 4)
        .vectorize(.j);

    const nest = comptime transformed_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{
        .{
            .num_blocks = 4,
            .block_size = 4,
            .idx_dim = 0,
        },
        .{
            .num_blocks = 4,
            .idx_dim = 0,
        },
        .{
            .num_blocks = 2,
            .block_size = 4,
            .idx_dim = 1,
        },
        .{
            .num_blocks = 1,
            .block_size = 4,
            .idx_dim = 1,
            .vector = true,
        },
    }, transformed_iter_space.loop_info);
    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{ .b = b });
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}
