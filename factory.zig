const std = @import("std");

pub fn GetChildType(comptime Type: type) type {
    return switch (@typeInfo(Type)) {
        inline else => |info| info.child,
    };
}

pub fn GetBaseType(comptime Type: type) type {
    return switch (@typeInfo(Type)) {
        .Vector => Type,
        .Int, .Float, .Bool => Type,
        inline else => |info| GetBaseType(info.child),
    };
}

pub fn GetScalarType(comptime Type: type) type {
    switch (@typeInfo(Type)) {
        .Pointer => |info| if (info.size == .Many) return GetScalarType(info.child),
        .Array => |info| return GetScalarType(info.child),
        .Int, .Float, .Bool => return Type,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Type}));
}

pub fn GetArrayType(comptime ScalarType: type, comptime ndims: u8, comptime shape: [ndims]?usize) type {
    var ArrayType = ScalarType;
    for (0..shape.len) |dim| {
        ArrayType = if (shape[ndims - dim - 1]) |s| [s]ArrayType else [*]ArrayType;
    }
    return ArrayType;
}

pub fn getNdims(comptime Array: type) u8 {
    switch (@typeInfo(Array)) {
        .Pointer => |info| if (info.size == .Many) return 1 + getNdims(info.child),
        .Array => |info| return 1 + getNdims(info.child),
        .Int, .Float, .Bool => return 0,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

pub fn getShape(comptime Array: type) [getNdims(Array)]?usize {
    switch (@typeInfo(Array)) {
        .Pointer => |info| if (info.size == .Many) return .{null} ++ getShape(info.child),
        .Array => |info| return .{info.len} ++ getShape(info.child),
        .Int, .Float, .Bool => return 0,
        else => {},
    }
    @compileError("Must provide an array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Array}));
}

// TODO: Load
// - Take in array, [ndims]usize index, and is last dim vectorized
// - Make a note if last dimension is vectorized

pub inline fn load(arr: anytype, idx: [getNdims(@TypeOf(arr.*))]usize) GetBaseType(@TypeOf(arr.*)) {
    var ptr: [*]const GetBaseType(@TypeOf(arr.*)) = &arr[0];
    inline for (idx) |i| {
        ptr = ptr[i..];
    }
    return ptr[0];
}

pub fn store(arr: anytype, idx: [getNdims(@TypeOf(arr.*))]usize, val: GetBaseType(@TypeOf(arr.*))) void {
    var ptr: [*]GetBaseType(@TypeOf(arr.*)) = &arr[0];
    inline for (idx) |i| {
        ptr = ptr[i..];
    }
    ptr[0] = val;
}

// Split:
//

fn validateInOutType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Array, .Pointer => {
            _ = getNdims(Type);
        },
        .Struct => |info| for (info.fields) |field| {
            _ = getNdims(field.type);
        },
        else => @compileError("Invalid input/output type"),
    }
}

fn PointerFields(comptime Type: type, const_ptr: bool) type {
    switch (@typeInfo(Type)) {
        .Array, .Pointer => {
            _ = getNdims(Type);
            return if (const_ptr) *const Type else *Type;
        },
        .Struct => |info| {
            var ptr_fields: [info.fields.len]std.builtin.Type.StructField = undefined;
            for (info.fields, 0..) |field, i| {
                _ = getNdims(field.type);
                const new_field: std.builtin.Type.StructField = .{
                    .type = if (const_ptr) *const field.type else *field.type,
                    .alignment = @alignOf(*field.type),
                    .name = field.name,
                    .is_comptime = false,
                    .default_value = null,
                };
                ptr_fields[i] = new_field;
            }
            return @Type(std.builtin.Type{
                .Struct = .{
                    .fields = &ptr_fields,
                    .layout = .auto,
                    .decls = &.{},
                    .is_tuple = false,
                },
            });
        },
        else => @compileError("Invalid input/output type"),
    }
}

fn IterationLogic(comptime In: type, comptime Out: type, comptime ndims: u8) type {
    validateInOutType(In);
    validateInOutType(Out);
    const nparams = @typeInfo(In).Struct.fields.len + @typeInfo(Out).Struct.fields.len + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;
    for (@typeInfo(In).Struct.fields) |field| {
        params[i] = .{ .is_generic = false, .is_noalias = false, .type = *const field.type };
        i += 1;
    }
    for (@typeInfo(Out).Struct.fields) |field| {
        params[i] = .{ .is_generic = false, .is_noalias = false, .type = *field.type };
        i += 1;
    }
    params[i] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [ndims]usize,
    };
    return @Type(.{ .Fn = .{
        .calling_convention = .Unspecified,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

const Nest = struct { loops: Loop, logic: *const anyopaque };

const Loop = struct {
    lower: usize = 0,
    upper: usize,

    // step: usize = 1,
    inner: union(enum) {
        loop: *const Loop,
        logic: *const anyopaque,
    },

    pub fn split(comptime loop: *const Loop, comptime In: type, comptime Out: type, comptime ndims: u8, split_count: usize) Loop {
        if (split_count == 1 or split_count == loop.upper - loop.lower + 1) {
            return loop.*;
        } else {
            return Loop{
                .lower = loop.lower,
                .upper = @divTrunc(loop.upper - loop.lower + 1, split_count),
                .inner = .{
                    .logic = struct {
                        fn iterationLogic(in: In, out: Out, idx: [ndims + 1]u8) void {
                            var new_idx =
                                nestEval(
                                &Loop{
                                    .upper = split_count,
                                    .inner = loop.inner,
                                },
                                In,
                                Out,
                                ndims,
                                in,
                                out,
                            );
                        }
                    }.iterationLogic,
                },

                // TODO: loop inner should be a new logic containing the index transform necessary to account for the axis split
                // e.g.
                // fn iterationLogic(in: anytype, out: anytype, idx: [4]usize) {
                //    i = idx[0];
                //    ii = idx[1];
                //    j = idx[2];
                //    k = idx[3];
                //    nestEval(&Loop{
                //        .upper = split_count,
                //        .inner = loop.inner,
                //    }, @TypeOf(in), @TypeOf(out), .{i * step_count + ii, j, k});
                // }
                // This way, the iteration logic provided by the user does not need to change but the loop transform still happens
                // .inner = &Loop{
                //     .upper = split_count,
                //     .inner = loop.inner,
                // },
            };
        }
    }

    inline fn baseEval(
        comptime loop: *const Loop,
        comptime In: type,
        comptime Out: type,
        comptime ndims: u8,
        in: anytype,
        out: anytype,
        base_idx: [ndims - 1]usize,
    ) void {
        var idx: [ndims]usize = base_idx ++ .{0};
        for (loop.lower..loop.upper) |loopvar| {
            idx[ndims - 1] = loopvar;
            @call(.auto, @as(*const IterationLogic(In, Out, ndims), @ptrCast(loop.inner.logic)), in ++ out ++ .{idx});
            // iterationLogic(in, out, idx);
        }
    }

    inline fn nestEval(
        comptime loop: *const Loop,
        comptime In: type,
        comptime Out: type,
        comptime ndims: u8,
        comptime depth: u8,
        in: anytype,
        out: anytype,
        base_idx: [depth + 1]usize,
    ) void {
        switch (loop.inner) {
            .loop => |inner| {
                var idx: [ndims]usize = .{0} ** ndims;
                inline for (0..depth + 1) |d| {
                    idx[d] = base_idx[d];
                }
                for (loop.lower..loop.upper) |loopvar| {
                    idx[depth] = loopvar;
                    nestEval(inner, In, Out, ndims, depth + 1, in, out, idx[0 .. depth + 2].*);
                }
            },
            .logic => baseEval(loop, In, Out, ndims, in, out, base_idx[0 .. ndims - 1].*),
        }
    }

    pub inline fn eval(
        comptime loop: *const Loop,
        comptime In: type,
        comptime Out: type,
        comptime ndims: u8,
        in: anytype,
        out: anytype,
    ) void {
        for (loop.lower..loop.upper) |loopvar| {
            nestEval(loop, In, Out, ndims, 0, in, out, .{loopvar});
        }
    }
};

pub fn MatMul(
    comptime M: usize,
    comptime N: usize,
    comptime P: usize,
    comptime dtype: type,
) type {
    return struct {
        const In: type = struct {
            A: [M][P]dtype,
            B: [P][N]dtype,
        };
        const Out: type = struct {
            C: [M][N]dtype,
        };
        fn iterationLogic(A: *const [M][P]dtype, B: *const [P][N]dtype, C: *[M][N]dtype, idx: [3]usize) void {
            const i = idx[0];
            const j = idx[1];
            const k = idx[2];
            const a = load(A, .{ i, k });
            const b = load(B, .{ k, j });
            const c = load(C, .{ i, j });
            store(C, .{ i, j }, a * b + c);
            std.debug.print("out.C[{[i]d}][{[j]d}] += in.A[{[i]d}][{[k]d}] * in.B[{[k]d}][{[j]d}]\r", .{ .i = i, .j = j, .k = k });
        }
        fn nest() Loop {
            return Loop{
                .upper = M,
                .inner = .{ .loop = &Loop{
                    .upper = N,
                    .inner = .{ .loop = &Loop{
                        .upper = P,
                        .inner = .{ .logic = iterationLogic },
                    } },
                } },
            };
        }
        pub fn eval(in: anytype, out: anytype) void {
            (comptime nest()).eval(In, Out, 3, in, out);
        }
    };
}

export fn main() void {
    const SGEMM = MatMul(16, 16, 16, f32);
    var A: [16][16]f32 = undefined;
    var B: [16][16]f32 = undefined;
    var C: [16][16]f32 = undefined;

    SGEMM.eval(.{ &A, &B }, .{&C});
}

test MatMul {
    const SGEMM = MatMul(16, 16, 16, f32);
    var A: [16][16]f32 = undefined;
    var B: [16][16]f32 = undefined;
    var C: [16][16]f32 = undefined;

    std.debug.print("\n", .{});
    SGEMM.eval(.{ &A, &B }, .{&C});
}
