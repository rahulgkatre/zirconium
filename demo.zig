const zirconium = @import("zirconium.zig");

const M = 1024;
const N = 1024;
const P = 1024;

const A = [M][P]f32;
const B = [N][P]f32;
const C = [M][N]f32;

const BLOCK_SIZE = 128;
const SIMD_SIZE = 8;

const Indices = enum { i, j, k };

const iter_space = zirconium.IterationSpace([M][N][P]f32, Indices)
    .init()
    .tile(&.{ .{ 0, BLOCK_SIZE }, .{ 1, BLOCK_SIZE } })
    .split(4, SIMD_SIZE)
    .parallel(1)
    .vectorize();

const Args = struct {
    a: A,
    b: B,
    c: C,
};

const matmul_logic: zirconium.Logic(Args, Indices) = struct {
    inline fn logic(
        a: *const zirconium.Buffer(A, Indices),
        b_t: *const zirconium.Buffer(B, Indices),
        c: *zirconium.Buffer(C, Indices),
        idx: [3]usize,
    ) void {
        const _a = a.load(.{ .i, .k }, idx);
        const _b = b_t.load(.{ .j, .k }, idx);
        const _c = c.load(.{ .i, .j }, idx);
        c.store(.{ .i, .j }, @reduce(.Add, _a * _b) + _c, idx);
    }
}.logic;

const nest = iter_space.nest(Args, matmul_logic);

pub fn main() !void {
    // @compileLog(@TypeOf(iter_space));
    // TODO: Need to support vectorized reduction
    const std = @import("std");
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = try nest.alloc(A, arena.allocator());
    const b = try nest.alloc(B, arena.allocator());
    const c = try nest.alloc(C, arena.allocator());
    nest.eval(.{ &a, &b, &c });
}
