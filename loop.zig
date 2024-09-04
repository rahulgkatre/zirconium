const std = @import("std");
const buffer = @import("buffer.zig");

pub const Nest = struct {
    pub const Loop = struct {
        lower: usize = 0,
        upper: usize,
        inner: ?*const Loop = null,

        step_size: usize = 1,
        unrolled: bool = false,
        vector: bool = false,
        block_info: buffer.BlockInfo,

        inline fn evalRouter(
            comptime loop: *const Loop,
            comptime Const: type,
            comptime Mutable: type,
            comptime ndims: u8,
            const_args: anytype,
            mut_args: anytype,
            base_idx: [ndims]usize,
            comptime logic: Logic(Const, Mutable, ndims),
        ) void {
            var idx = base_idx;
            const base_for_dim = base_idx[loop.block_info.orig_dim];
            var loopvar = loop.lower;

            if (comptime loop.vector) {
                const vectorized_logic = comptime @as(*const VectorizedLogic(Const, Mutable, ndims), @ptrCast(&logic));
                // reinterpret the args as their corresponding vectorized versions
                // this is always valid as VectorizedLogic would catch invalid types for Const and Mutable
                const vectorized_logic_args = @as(*const std.meta.ArgsTuple(@typeInfo(@TypeOf(vectorized_logic)).Pointer.child), @ptrCast(&(const_args ++ mut_args ++ .{idx}))).*;
                @call(.always_inline, vectorized_logic, vectorized_logic_args);
            } else if (comptime loop.unrolled) {
                inline while (loopvar < loop.upper) : (loopvar += loop.step_size) {
                    idx[loop.block_info.orig_dim] += loop.block_info.block_size * loopvar;
                    // TODO: Replace inner with a []const union(logic|loop) and iterate over that slice here
                    if (comptime loop.inner) |inner|
                        inner.evalRouter(Const, Mutable, ndims, const_args, mut_args, idx, logic)
                    else
                        @call(.always_inline, logic, const_args ++ mut_args ++ .{idx});
                    idx[loop.block_info.orig_dim] = base_for_dim;
                }
            } else {
                while (loopvar < loop.upper) : (loopvar += loop.step_size) {
                    idx[loop.block_info.orig_dim] += loop.block_info.block_size * loopvar;
                    if (comptime loop.inner) |inner|
                        inner.evalRouter(Const, Mutable, ndims, const_args, mut_args, idx, logic)
                    else
                        @call(.always_inline, logic, const_args ++ mut_args ++ .{idx});
                    idx[loop.block_info.orig_dim] = base_for_dim;
                }
            }
        }
    };

    loop: ?*const Loop,
    orig_ndims: u8,
    float_mode: std.builtin.FloatMode,

    pub inline fn eval(
        comptime nest: *const Nest,
        comptime Const: type,
        comptime Mutable: type,
        // TODO: Restrict these types
        const_args: anytype,
        mut_args: anytype,
        comptime logic: Logic(Const, Mutable, nest.orig_ndims),
    ) void {
        @setFloatMode(nest.float_mode);

        if (nest.loop) |loop| {
            const idx: [nest.orig_ndims]usize = .{0} ** nest.orig_ndims;
            loop.evalRouter(Const, Mutable, nest.orig_ndims, const_args, mut_args, idx, logic);
        } else {
            if (comptime Const != void) {
                @call(.always_inline, logic, const_args ++ mut_args ++ .{&(.{0} ** nest.orig_ndims)});
            } else {
                @call(.always_inline, logic, mut_args ++ .{&(.{0} ** nest.orig_ndims)});
            }
        }
    }
};

fn validateArgsType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Struct => {},
        else => @compileError("Invalid input/output type"),
    }
}

pub fn Logic(comptime Const: type, comptime Mutable: type, comptime ndims: u8) type {
    if (Const != void) validateArgsType(Const);
    validateArgsType(Mutable);

    const nparams = (if (Const != void) @typeInfo(Const).Struct.fields.len else 0) + @typeInfo(Mutable).Struct.fields.len + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;

    if (Const != void) {
        for (@typeInfo(Const).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *const buffer.AllocatedBuffer(field.type) };
            i += 1;
        }
    }
    for (@typeInfo(Mutable).Struct.fields) |field| {
        params[i] = .{ .is_generic = false, .is_noalias = false, .type = *buffer.AllocatedBuffer(field.type) };
        i += 1;
    }
    params[i] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [ndims]usize,
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}

pub fn VectorizedLogic(comptime Const: type, comptime Mutable: type, comptime ndims: u8) type {
    if (Const != void) validateArgsType(Const);
    validateArgsType(Mutable);

    const nparams = (if (Const != void) @typeInfo(Const).Struct.fields.len else 0) + @typeInfo(Mutable).Struct.fields.len + 1;
    var params: [nparams]std.builtin.Type.Fn.Param = undefined;
    var i = 0;

    if (Const != void) {
        for (@typeInfo(Const).Struct.fields) |field| {
            params[i] = .{ .is_generic = false, .is_noalias = false, .type = *const buffer.AllocatedBuffer(buffer.Vectorized(field.type)) };
            i += 1;
        }
    }
    for (@typeInfo(Mutable).Struct.fields) |field| {
        params[i] = .{ .is_generic = false, .is_noalias = false, .type = *buffer.AllocatedBuffer(buffer.Vectorized(field.type)) };
        i += 1;
    }
    params[i] = .{
        .is_generic = false,
        .is_noalias = false,
        .type = [ndims]usize,
    };
    return @Type(.{ .Fn = .{
        .calling_convention = std.builtin.CallingConvention.Inline,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}
