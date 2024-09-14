const std = @import("std");
const utils = @import("utils.zig");
const IterSpace = @import("IterSpace.zig");
const Buffer = @import("buffer.zig").Buffer;
const IterSpaceBuffer = @import("buffer.zig").IterSpaceBuffer;

const TiledBuffer = @import("buffer.zig").IndexedTiledBuffer;

fn validateArgsType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Struct => {},
        else => @compileError("Invalid input/output type"),
    }
}

pub fn LogicArgs(comptime Args: type, comptime Indices: type) type {
    var fields: [@typeInfo(Args).Struct.fields.len]std.builtin.Type.StructField = undefined;

    for (@typeInfo(Args).Struct.fields, 0..) |field, i| {
        const Type = Buffer(field.type, Indices);
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
            .layout = .auto,
        },
    });
}

/// When defining loop logic, use this type.
pub fn Logic(comptime Args: type, comptime Indices: type) type {
    const idx_ndims = @typeInfo(Indices).Enum.fields.len;
    if (Args != void) validateArgsType(Args);

    const params: [2]std.builtin.Type.Fn.Param = .{
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = LogicArgs(Args, Indices),
        },
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = [idx_ndims]usize,
        },
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

pub fn IterSpaceLogicArgs(comptime Args: type, comptime iter_space: IterSpace) type {
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
            .layout = .auto,
        },
    });
}

/// This type is used to specify the iteration space a buffer will be accessed through.
/// The iteration space and indices define memory access patterns.
pub fn IterSpaceLogic(comptime Args: type, comptime iter_space: IterSpace) type {
    if (Args != void) validateArgsType(Args);
    const params: [2]std.builtin.Type.Fn.Param = .{
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = IterSpaceLogicArgs(Args, iter_space),
        },
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = [iter_space.numIndices()]usize,
        },
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

pub fn ExternFnArgs(comptime Args: type, comptime iter_space: IterSpace) type {
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
pub fn ExternFn(comptime Args: type, comptime iter_space: IterSpace) type {
    if (Args != void) validateArgsType(Args);
    const params: [2]std.builtin.Type.Fn.Param = .{
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = ExternFnArgs(Args, iter_space),
        },
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = [iter_space.numIndices()]usize,
        },
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.C,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}
