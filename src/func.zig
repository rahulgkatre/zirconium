const std = @import("std");
const utils = @import("utils.zig");
const IterSpace = @import("IterSpace.zig");
const Buffer = @import("buffer.zig").Buffer;
const IterSpaceBuffer = @import("buffer.zig").IterSpaceBuffer;

const TiledBuffer = @import("buffer.zig").IndexedTiledBuffer;

pub fn validateArgsType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Struct => {},
        else => @compileError("Invalid input/output type"),
    }
}

/// When defining loop logic, use this type.
pub fn Func(comptime Args: type, comptime DataIndex: type) type {
    if (Args != void) validateArgsType(Args);
    const idx_ndims = @typeInfo(DataIndex).Enum.fields.len;
    var params: [@typeInfo(Args).Struct.fields.len + 1]std.builtin.Type.Fn.Param = undefined;
    params[0] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [idx_ndims]usize,
    };

    for (@typeInfo(Args).Struct.fields, 1..) |field, i| {
        params[i] = std.builtin.Type.Fn.Param{
            .is_generic = false,
            .is_noalias = false,
            .type = Buffer(field.type, DataIndex),
        };
    }

    const _Def = @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });

    return struct {
        pub const Def = _Def;
        pub fn Param(comptime Array: type) type {
            return Buffer(Array, DataIndex);
        }
    };
}

pub fn IterSpaceArgsTuple(comptime Args: type, comptime iter_space: IterSpace) type {
    var fields: [@typeInfo(Args).Struct.fields.len]std.builtin.Type.StructField = undefined;

    for (@typeInfo(Args).Struct.fields, 0..) |field, i| {
        const Type = IterSpaceBuffer(field.type, iter_space);
        fields[i] = std.builtin.Type.StructField{
            .alignment = @alignOf(Type),
            .is_comptime = false,
            .default_value = null,
            .name = std.fmt.comptimePrint("{d}", .{i}),
            .type = Type,
        };
    }

    return @Type(.{
        .Struct = std.builtin.Type.Struct{
            .decls = &.{},
            .fields = &fields,
            .is_tuple = true,
            .layout = .auto,
        },
    });
}

pub fn IterSpaceSeparateArgs(comptime Args: type, comptime iter_space: IterSpace) type {
    const idx_ndims = iter_space.numDataIndices();
    var params: [@typeInfo(Args).Struct.fields.len + 1]std.builtin.Type.Fn.Param = undefined;
    params[0] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [idx_ndims]usize,
    };

    for (@typeInfo(Args).Struct.fields, 1..) |field, i| {
        params[i] = std.builtin.Type.Fn.Param{
            .is_generic = false,
            .is_noalias = false,
            .type = IterSpaceBuffer(field.type, iter_space),
        };
    }

    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

pub fn IterSpaceKernel(comptime Args: type, comptime iter_space: IterSpace) type {
    const idx_ndims = iter_space.numDataIndices();
    var params: [@typeInfo(Args).Struct.fields.len + 1]std.builtin.Type.Fn.Param = undefined;
    params[0] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [idx_ndims]usize,
    };

    for (@typeInfo(Args).Struct.fields, 1..) |field, i| {
        params[i] = std.builtin.Type.Fn.Param{
            .is_generic = false,
            .is_noalias = false,
            .type = IterSpaceBuffer(field.type, iter_space),
        };
    }

    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Kernel,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

/// This type is used to specify the iteration space a buffer will be accessed through.
/// The iteration space and indices define memory access patterns.
pub fn IterSpaceFunc(comptime Args: type, comptime iter_space: IterSpace) type {
    if (Args != void) validateArgsType(Args);
    return struct {
        pub const TupleFn = @Type(.{ .Fn = .{
            .calling_convention = std.builtin.CallingConvention.Inline,
            .is_generic = false,
            .is_var_args = false,
            .params = &.{
                .{
                    .is_generic = false,
                    .is_noalias = false,
                    .type = [iter_space.numDataIndices()]usize,
                },
                .{
                    .is_generic = false,
                    .is_noalias = false,
                    .type = IterSpaceArgsTuple(Args, iter_space),
                },
            },
            .return_type = void,
        } });
        pub const SeparatesFn = IterSpaceSeparateArgs(Args, iter_space);
    };
}

pub fn ExternFuncParam(comptime Args: type, comptime iter_space: IterSpace) type {
    var fields: [@typeInfo(Args).Struct.fields.len]std.builtin.Type.StructField = undefined;

    for (@typeInfo(Args).Struct.fields, 0..) |field, i| {
        const Type = IterSpaceBuffer(field.type, iter_space);
        fields[i] = std.builtin.Type.StructField{
            .alignment = @alignOf(Type),
            .is_comptime = false,
            .default_value = null,
            .name = field.name,
            .type = Type,
        };
    }

    return @Type(.{
        .Struct = std.builtin.Type.Struct{
            .decls = &.{},
            .fields = &fields,
            .is_tuple = false,
            .layout = .@"extern",
        },
    });
}

/// This type is used to specify the iteration space a buffer will be accessed through.
/// The iteration space and indices define memory access patterns.
pub fn ExternFunc(comptime Args: type, comptime iter_space: IterSpace) type {
    if (Args != void) validateArgsType(Args);
    const params: [1]std.builtin.Type.Fn.Param = .{
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = ExternFuncParam(Args, iter_space),
        },
    };
    const _Def = @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.C,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
    return struct {
        pub const Def = _Def;
        pub const Param = ExternFuncParam(Args, iter_space);
    };
}
