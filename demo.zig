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
    .init()
    .tile(&.{ .{ 0, BLOCK_SIZE }, .{ 1, BLOCK_SIZE } })
    .split(4, SIMD_SIZE)
    .vectorize();

const Indices: type = iter_space.Indices(.{ "i", "j", "k" });

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
        c.store(.{ .i, .j }, @mulAdd(@TypeOf(a.*).Unit, _a, _b, _c), idx);
    }
}.logic;

const nest = iter_space.nest(Args, .{ "i", "j", "k" }, matmul_logic);

pub fn main() !void {
    // @compileLog(@TypeOf(iter_space));
    // TODO: What is wrong with alignment here?
    const std = @import("std");
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = try zirconium.Buffer(A, Indices).alloc(arena.allocator());
    const b = try zirconium.Buffer(B, Indices).alloc(arena.allocator());
    const c = try zirconium.Buffer(C, Indices).alloc(arena.allocator());
    nest.eval(.{ &a, &b }, .{&c});
}
