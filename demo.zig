const zirconium = @import("zirconium.zig");

// First, define the constants: sizes and types of the matrices
const M = 1024;
const N = 1024;
const P = 1024;

const BLOCK_SIZE = 128;
const SIMD_SIZE = 8;

const A = [M][P]f32;
const B = [N][P]f32;
const C = [M][N]f32;

// The args to matmul, all layout information will be provided
// by the buffer that wraps around these types
const Args = struct {
    a: A,
    b: B,
    c: C,
};

// Next, define the base indices for the loop nest.
// It is better to use single character index names,
// as splitting / tiling will repeat the character for each level of split.
const DataIndex = enum { i, j, k };

// Use these indices to create the iteration space, and transform it.
// This does not have to be done in a functional/chained way as the type does not change.
// The actual loopvar enum type can be accessed through iter_space.LoopVar
const iter_space = zirconium.IterSpace
    .init([M][N][P]f32, DataIndex)
    .tile(&.{ .{ .i, BLOCK_SIZE }, .{ .j, BLOCK_SIZE } })
    .split(.k, SIMD_SIZE)
    .parallel(.jj, null)
    .vectorize(.k);

// The innermost logic for matmul
const Func = zirconium.Func(Args, DataIndex);
const matmul_func: Func.Def = struct {
    pub inline fn logic(
        idx: [3]usize,
        a: Func.Param(A),
        b: Func.Param(B),
        c: Func.Param(C),
    ) void {
        const _a = a.load(.{ .i, .k }, idx);
        const _b = b.load(.{ .j, .k }, idx);
        const _c = c.load(.{ .i, .j }, idx);
        c.store(.{ .i, .j }, @reduce(.Add, _a * _b) + _c, idx);
    }
}.logic;

// Convert into a nest
const nest = iter_space.nest(Args, matmul_func);

// Build into an export (C ABI function)
export const matmul = nest.buildExtern();

// Optional: run the function
// Can also just export .so and call from any language that supports C ABI
pub fn main() !void {
    const std = @import("std");
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = try nest.alloc(A, arena.allocator());
    const b = try nest.alloc(B, arena.allocator());
    const c = try nest.alloc(C, arena.allocator());
    matmul(.{ .a = a, .b = b, .c = c });
}
