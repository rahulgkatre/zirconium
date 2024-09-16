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

fn FuncArgs(comptime Args: type, comptime DataIndex: type) type {
    var fields: [@typeInfo(Args).Struct.fields.len]std.builtin.Type.StructField = undefined;

    for (@typeInfo(Args).Struct.fields, 0..) |field, i| {
        const Type = Buffer(field.type, DataIndex);
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

fn FuncDef(comptime Args: type, comptime DataIndex: type) type {
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

    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

/// When defining loop logic, use this type.
pub fn Func(comptime Args: type, comptime DataIndex: type) type {
    if (Args != void) validateArgsType(Args);
    return struct {
        pub const Def = FuncDef(Args, DataIndex);
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

pub fn IterSpaceArgsStruct(comptime Args: type, comptime iter_space: IterSpace) type {
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
            .is_tuple = false,
            .layout = .auto,
        },
    });
}

fn IterSpaceFuncSeparatesFn(comptime Args: type, comptime iter_space: IterSpace) type {
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

/// This type is used to specify the iteration space a buffer will be accessed through.
/// The iteration space and indices define memory access patterns.
pub fn IterSpaceFunc(comptime Args: type, comptime iter_space: IterSpace) type {
    if (Args != void) validateArgsType(Args);
    return struct {
        pub const StructFn = @Type(.{ .Fn = .{
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
                    .type = IterSpaceArgsStruct(Args, iter_space),
                },
            },
            .return_type = void,
        } });
        pub const SeparatesFn = IterSpaceFuncSeparatesFn(Args, iter_space);
    };
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
    const params: [1]std.builtin.Type.Fn.Param = .{
        .{
            .is_generic = false,
            .is_noalias = false,
            .type = ExternFnArgs(Args, iter_space),
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
