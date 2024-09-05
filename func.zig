const std = @import("std");
const utils = @import("utils.zig");
const AllocatedBuffer = @import("buffer.zig").AllocatedBuffer;
const AllocatedVecBuffer = @import("buffer.zig").AllocatedVecBuffer;

fn validateArgsType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Struct => {},
        else => @compileError("Invalid input/output type"),
    }
}

pub fn externFnParams(comptime In: type, comptime InOut: type) []std.builtin.Type.Fn.Param {
    if (In != void) validateArgsType(In);
    if (InOut != void) validateArgsType(InOut);

    const nparams = (if (In != void) @typeInfo(In).Struct.fields.len else 0) + (if (InOut != void) @typeInfo(InOut).Struct.fields.len else 0);
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;
    // var fields: [nparams]std.builtin.Type.StructField = undefined;
    if (In != void) {
        for (@typeInfo(In).Struct.fields) |field| {
            const Type = AllocatedBuffer(field.type);
            // const alignment: usize = Type.alignment;
            // fields[i] = std.builtin.Type.StructField{
            //     .alignment = alignment,
            //     .type = std.meta.FieldType(Type, .raw),
            //     .name = field.name,
            // };
            params[i] = std.builtin.Type.Fn.Param{
                .type = std.meta.FieldType(Type, .raw),
            };
            i += 1;
        }
    }
    if (InOut != void) {
        for (@typeInfo(In).Struct.fields) |field| {
            const Type = AllocatedBuffer(field.type);
            // const alignment: usize = Type.alignment;
            // fields[i] = std.builtin.Type.StructField{
            //     .alignment = alignment,
            //     .type = std.meta.FieldType(Type, .raw),
            //     .name = field.name,
            // };
            params[i] = std.builtin.Type.Fn.Param{
                .type = std.meta.FieldType(Type, .raw),
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

pub fn ExternFn(comptime In: type, comptime InOut: type) type {
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.C,
        .is_generic = false,
        .is_var_args = false,
        .params = externFnParams(In, InOut),
        .return_type = void,
    } });
}

pub fn IterationLogic(comptime In: type, comptime InOut: type, comptime idx_ndims: u8) type {
    if (In != void) validateArgsType(In);
    if (InOut != void) validateArgsType(InOut);

    const nparams = (if (In != void) @typeInfo(In).Struct.fields.len else 0) + (if (InOut != void) @typeInfo(InOut).Struct.fields.len else 0) + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;

    if (In != void) {
        for (@typeInfo(In).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *const AllocatedBuffer(field.type) };
            i += 1;
        }
    }
    if (InOut != void) {
        for (@typeInfo(InOut).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *AllocatedBuffer(field.type) };
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

pub fn VectorizedLogic(comptime In: type, comptime InOut: type, comptime idx_ndims: u8, comptime vec_len: usize) type {
    if (In != void) validateArgsType(In);
    if (InOut != void) validateArgsType(InOut);

    const nparams = (if (In != void) @typeInfo(In).Struct.fields.len else 0) + (if (InOut != void) @typeInfo(InOut).Struct.fields.len else 0) + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;

    if (In != void) {
        for (@typeInfo(In).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *const AllocatedVecBuffer(field.type, vec_len) };
            i += 1;
        }
    }
    if (InOut != void) {
        for (@typeInfo(InOut).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *AllocatedVecBuffer(field.type, vec_len) };
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
