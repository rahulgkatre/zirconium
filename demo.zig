const zirconium = @import("zirconium.zig");

const M = 1024;
const N = 1024;
const P = 512;

const A = [M][P]f32;
const B = [P][N]f32;
const C = [M][N]f32;

const BLOCK_SIZE = 128;
const SIMD_SIZE = 8;

const iter_space = zirconium.IterationSpace([M][N][P]f32)
    .init(.{ "dim0", "dim1", "dim2" })
    .tile(&.{ .{ 0, BLOCK_SIZE }, .{ 1, BLOCK_SIZE } })
    .split(4, SIMD_SIZE)
    .vectorize();

const Indices = iter_space.Indices();

const Args = struct {
    a: A,
    b: B,
    c: C,
};

const matmul_logic: zirconium.Logic(Args, Indices) = struct {
    inline fn logic(
        a: *const zirconium.Buffer(A),
        b_t: *const zirconium.Buffer(B),
        c: *zirconium.Buffer(C),
        idx: [3]usize,
    ) void {
        const _a = a.load(.{ .dim0, .dim1 }, idx);
        const _b = b_t.load(.{ .dim1, .dim2 }, idx);
        const _c = c.load(.{ .dim0, .dim1 }, idx);
        c.store(.{ .dim0, .dim1 }, @mulAdd(@TypeOf(a.*).Unit, _a, _b, _c), idx);
    }
}.logic;

const nest = iter_space.nest(Args, matmul_logic);

pub fn main() !void {
    // @compileLog(@TypeOf(iter_space));
    // TODO: What is wrong with alignment here?
    const std = @import("std");
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = try zirconium.Buffer(A).alloc(arena.allocator());
    const b = try zirconium.Buffer(B).alloc(arena.allocator());
    const c = try zirconium.Buffer(C).alloc(arena.allocator());
    nest.eval(.{ &a, &b }, .{&c});
}
