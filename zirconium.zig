const buffer = @import("buffer.zig");
const loop = @import("loop.zig");
const iterspace = @import("iterspace.zig");

const M = 1024;
const N = 1024;
const P = 512;

const A = [M][P]f32;
const B = [P][N]f32;
const C = [M][N]f32;

const In = struct {
    a: buffer.AllocatedBuffer(A),
    b: buffer.AllocatedBuffer(B),
};

const InOut = struct {
    c: buffer.AllocatedBuffer(C.Arr),
};

const matmul_logic: loop.Logic(In, InOut, 3) = struct {
    inline fn logic(
        a: buffer.AllocatedBuffer(A.Arr),
        b: buffer.AllocatedBuffer(B.Arr),
        c: buffer.AllocatedBuffer(C.Arr),
        idx: []const usize,
    ) void {
        const _a = a.load(.{ idx[0], idx[2] });
        const _b = b.load(.{ idx[2], idx[1] });
        const _c = c.load(.{ idx[0], idx[1] });
        c.store(_a * _b + _c, .{ idx[0], idx[1] });
    }
}.logic;
