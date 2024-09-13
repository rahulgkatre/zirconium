const std = @import("std");
const utils = @import("utils.zig");
const Buffer = @import("buffer.zig").Buffer;
const IterSpaceBuffer = @import("buffer.zig").IterSpaceBuffer;

const TiledBuffer = @import("buffer.zig").IndexedTiledBuffer;

fn validateArgsType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Struct => {},
        else => @compileError("Invalid input/output type"),
    }
}

pub fn externFnParams(comptime Args: type) []std.builtin.Type.Fn.Param {
    if (Args != void) validateArgsType(Args);

    const nparams = (if (Args != void) @typeInfo(Args).Struct.fields.len else 0);
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;
    // var fields: [nparams]std.builtin.Type.StructField = undefined;
    if (Args != void) {
        for (@typeInfo(Args).Struct.fields) |field| {
            const Type = Buffer(field.type);
            // const alignment: usize = Type.alignment;
            // fields[i] = std.builtin.Type.StructField{
            //     .alignment = alignment,
            //     .type = std.meta.FieldType(Type, .data),
            //     .name = field.name,
            // };
            params[i] = std.builtin.Type.Fn.Param{
                .type = std.meta.FieldType(Type, .data),
            };
            i += 1;
        }
    }
    // const argType: type = @Type(std.builtin.Type{ .Struct = .{
    //     .fields = &fields,
    //     .decls = &.{},
    //     .layout = .@"extern",
    // } });
    // params[0] = .{ .is_generic = false, .is_noalias = false, .type = argType };
    return &params;
}

pub fn ExternFn(comptime Args: type) type {
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.C,
        .is_generic = false,
        .is_var_args = false,
        .params = externFnParams(Args),
        .return_type = void,
    } });
}

/// When defining loop logic, use this type.
pub fn Logic(comptime Args: type, comptime Indices: type) type {
    const idx_ndims = @typeInfo(Indices).Enum.fields.len;
    if (Args != void) validateArgsType(Args);

    const nparams = (if (Args != void) @typeInfo(Args).Struct.fields.len else 0) + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;

    if (Args != void) {
        for (@typeInfo(Args).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *Buffer(field.type, Indices) };
            i += 1;
        }
    }
    params[i] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [idx_ndims]usize,
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

/// This type is used to specify the iteration space a buffer will be accessed through.
/// The iteration space and indices define memory access patterns.
pub fn IterSpaceLogic(comptime Args: type, comptime iter_space: anytype) type {
    if (Args != void) validateArgsType(Args);

    const nparams = (if (Args != void) @typeInfo(Args).Struct.fields.len else 0) + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;

    if (Args != void) {
        for (@typeInfo(Args).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *IterSpaceBuffer(field.type, iter_space) };
            i += 1;
        }
    }
    params[i] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [iter_space.idx_ndims]usize,
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

// pub fn VectorizedLogic(comptime Args: type, comptime iter_space: anytype) type {
//     const vec_len = iter_space.shape[]
//     if (Args != void) validateArgsType(Args);

//     const nparams = (if (Args != void) @typeInfo(Args).Struct.fields.len else 0) + 1;
//     var params: [nparams]std.builtin.Type.Fn.Param = undefined;
//     var i = 0;

//     if (Args != void) {
//         for (@typeInfo(Args).Struct.fields) |field| {
//             params[i] = .{ .is_generic = false, .is_noalias = false, .type = *TiledBuffer(field.type, @Vector(vec_len, utils.Datatype(field.type))) };
//             i += 1;
//         }
//     }
//     params[i] = .{
//         .is_generic = false,
//         .is_noalias = false,
//         .type = [idx_ndims]usize,
//     };
//     return @Type(.{ .Fn = .{
//         .calling_convention = std.builtin.CallingConvention.Inline,
//         .is_generic = false,
//         .is_var_args = false,
//         .params = &params,
//         .return_type = void,
//     } });
// }
