pub const Buffer = @import("src/buffer.zig").Buffer;
pub const Layout = @import("src/buffer.zig").Layout;
pub const IterSpace = @import("src/IterSpace.zig");
pub const Func = @import("src/func.zig").Func;

const zirconium = @This();
test zirconium {
    _ = @import("src/loop.zig");
}
