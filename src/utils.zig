const std = @import("std");
const IterationSpace = @import("iterspace.zig").IterationSpace;

pub const IndexInfo = struct {
    orig_dim: u8,
    num_blocks: usize,
    block_size: usize,
    vector: bool,
};

pub fn ShapeToArray(comptime dtype: type, comptime ndims: u8, comptime shape: *const [ndims]usize) type {
    var Type = dtype;
    for (0..shape.len) |dim| {
        const s = shape[ndims - dim - 1];
        Type = [s]Type;
    }
    return Type;
}

pub fn Datatype(comptime Array: type) type {
    switch (@typeInfo(Array)) {
        .Vector => |info| return info.child,
        .Pointer => |info| if (info.size == .Many) return IterationSpace(info.child).dtype,
        .Array => |info| return IterationSpace(info.child).dtype,
        .Int, .Float, .Bool => return Array,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

pub fn extractNdims(comptime Array: type) u8 {
    switch (@typeInfo(Array)) {
        .Pointer => |info| if (info.size == .Many) return 1 + IterationSpace(info.child).ndims,
        .Array => |info| return 1 + IterationSpace(info.child).ndims,
        .Int, .Float, .Bool => return 0,
        .Vector => return 1,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

pub fn extractShape(comptime Array: type) [extractNdims(Array)]usize {
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

pub fn Unit(comptime Array: type) type {
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

pub fn arrayPermute(comptime T: type, comptime len: u8, array: [len]T, comptime perm: [len]u8) [len]T {
    var used: [len]bool = [_]bool{false} ** len;
    for (perm) |p| {
        if (p < len and !used[p]) {
            used[p] = true;
        } else {
            @compileError(std.fmt.comptimePrint("Invalid permutation {any}", .{perm}));
        }
    }
    for (used) |u| {
        if (!u) @compileError("Not all dims in permutation were used");
    }
    var new_array: [len]T = undefined;
    for (0..len) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}
