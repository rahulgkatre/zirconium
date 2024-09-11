const zirconium = @import("zirconium.zig");

const M = 1024;
const N = 1024;
const P = 512;

const A = [M][P]f32;
const B = [P][N]f32;
const C = [M][N]f32;

const BLOCK_SIZE = 128;
const SIMD_SIZE = 8;

const Args = struct {
    a: A,
    b: B,
    c: C,
};

const matmul_logic: zirconium.IterationLogic(Args, 3) = struct {
    inline fn logic(
        a: *const zirconium.AllocatedBuffer(A),
        b: *const zirconium.AllocatedBuffer(B),
        c: *zirconium.AllocatedBuffer(C),
        idx: [3]usize,
    ) void {
        const _a = a.load(.{ idx[0], idx[2] });
        const _b = b.load(.{ idx[2], idx[1] });
        const _c = c.load(.{ idx[0], idx[1] });
        c.store(_a * _b + _c, .{ idx[0], idx[1] });
    }
}.logic;

const iter_space = zirconium.IterationSpace([M][N][P]f32)
    .init()
    .tile(&.{ .{ 0, BLOCK_SIZE }, .{ 1, BLOCK_SIZE } })
    .split(4, SIMD_SIZE)
    .vectorize();

const nest = iter_space.nest(Args, matmul_logic);

pub fn main() !void {
    @compileLog(@TypeOf(iter_space));
    // TODO: What is wrong with alignment here?
    const std = @import("std");
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = try zirconium.AllocatedBuffer(A).alloc(arena.allocator());
    const b = try zirconium.AllocatedBuffer(B).alloc(arena.allocator());
    const c = try zirconium.AllocatedBuffer(C).alloc(arena.allocator());
    nest.eval(.{ &a, &b }, .{&c});
}
