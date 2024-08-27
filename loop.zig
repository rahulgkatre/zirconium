const std = @import("std");
fn validateInOutType(comptime Type: type) void {
    switch (@typeInfo(Type)) {
        .Struct => {},
        else => @compileError("Invalid input/output type"),
    }
}

pub const Nest = struct {
    loop: ?*const Loop,

    pub inline fn eval(
        nest: *const Nest,
        comptime In: type,
        comptime Out: type,
        comptime ndims: u8,
        in: anytype,
        out: anytype,
        comptime logic: Logic(In, Out),
    ) void {
        if (nest.loop) |loop| {
            loop.eval(In, Out, ndims, in, out, logic);
        } else {
            @call(.always_inline, logic, in ++ out ++ .{&(.{0} ** ndims)});
        }
    }
};

pub fn Logic(comptime In: type, comptime Out: type) type {
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
        .type = []const usize,
    };
    return @Type(.{ .Fn = .{
        .calling_convention = .kernel,
        .is_generic = false,
        .is_var_args = false,
        .params = &params,
        .return_type = void,
    } });
}
const Callable = *const anyopaque;

pub const Loop = struct {
    lower: usize = 0,
    upper: usize,
    inner: ?*const Loop = null,

    inline fn nestEval(
        loop: *const Loop,
        comptime In: type,
        comptime Out: type,
        comptime ndims: u8,
        comptime depth: u8,
        in: anytype,
        out: anytype,
        base_idx: []usize,
        comptime logic: *const Logic(In, Out),
    ) void {
        if (comptime depth < ndims) {
            var idx = base_idx;
            if (loop.inner) |inner| {
                for (loop.lower..loop.upper) |loopvar| {
                    idx[depth] = loopvar;
                    inner.nestEval(In, Out, ndims, depth + 1, in, out, idx, logic);
                }
            } else {
                for (loop.lower..loop.upper) |loopvar| {
                    idx[depth] = loopvar;
                    @call(.always_inline, logic, in ++ out ++ .{idx});
                }
            }
            idx[depth] = 0;
        }
    }

    pub inline fn eval(
        loop: *const Loop,
        comptime In: type,
        comptime Out: type,
        comptime ndims: u8,
        in: anytype,
        out: anytype,
        logic: *const Logic(In, Out),
    ) void {
        var idx: [ndims]usize = .{0} ** ndims;
        nestEval(loop, In, Out, ndims, 0, in, out, @constCast(&idx), logic);
    }
};
