//! Worker-local native OpenGL ownership. GPU objects must be deleted before this
//! owner is dropped; neither this owner nor its GL dispatch table may be moved
//! to another thread.

use std::marker::PhantomData;
use std::rc::Rc;

use anr_core::Result;
#[cfg(any(target_os = "windows", target_os = "linux"))]
use glow::HasContext;

#[cfg(any(target_os = "windows", target_os = "linux"))]
use glutin::prelude::*;

pub(crate) struct GlContext {
    pub(crate) gl: glow::Context,
    // Fields are dropped in declaration order: dispatch table, context/surface,
    // native window. Rc prevents Send and Sync even if dependency types change.
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    _native: native::NativeContext,
    _thread_local: PhantomData<Rc<()>>,
}

impl GlContext {
    pub(crate) fn new() -> Result<Self> {
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            use std::ffi::CString;

            let native = native::NativeContext::new()?;
            // WGL exposes modern entry points only after make_current. Reject
            // missing essential functions before glow can try calling them.
            for name in [
                "glGetString",
                "glGetStringi",
                "glGetIntegerv",
                "glGetInteger64v",
                "glGetIntegeri_v",
                "glDispatchCompute",
                "glMemoryBarrier",
                "glBindBufferBase",
            ] {
                let symbol = CString::new(name).expect("literal GL symbol has no NUL");
                if native.proc_address(&symbol).is_null() {
                    return Err(crate::backend(format!(
                        "缺少 OpenGL 4.3 Compute 入口：{name}"
                    )));
                }
            }
            // SAFETY: native owns a current context on this thread and remains
            // alive until after the dispatch table is dropped.
            let gl = unsafe {
                glow::Context::from_loader_function(|name| {
                    let symbol = CString::new(name).expect("GL symbol has no NUL");
                    native.proc_address(&symbol)
                })
            };
            // Keep the complete RAII owner before any fallible capability
            // queries, so failure also unbinds and destroys the current context.
            let owner = Self {
                gl,
                _native: native,
                _thread_local: PhantomData,
            };
            owner.validate()?;
            Ok(owner)
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            Err(crate::backend(
                "OpenGL Compute 后端只支持 Windows/WGL 和 Linux/EGL；macOS 原生 OpenGL 不支持 4.3 Compute",
            ))
        }
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    fn validate(&self) -> Result<()> {
        let version = self.gl.version();
        if version.is_embedded || (version.major, version.minor) < (4, 3) {
            return Err(crate::backend(format!(
                "需要桌面 OpenGL 4.3 Core，驱动提供 {}.{}{}",
                version.major,
                version.minor,
                if version.is_embedded { " ES" } else { "" },
            )));
        }
        // SAFETY: the context is current here. These queries are core in 4.3.
        unsafe {
            if self.gl.get_parameter_i32(glow::CONTEXT_PROFILE_MASK)
                & glow::CONTEXT_CORE_PROFILE_BIT as i32
                == 0
            {
                return Err(crate::backend("驱动没有创建 OpenGL Core profile 上下文"));
            }
            for (name, parameter, required) in [
                ("SSBO bindings", glow::MAX_SHADER_STORAGE_BUFFER_BINDINGS, 4),
                (
                    "compute SSBO blocks",
                    glow::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
                    4,
                ),
                (
                    "combined SSBO blocks",
                    glow::MAX_COMBINED_SHADER_STORAGE_BLOCKS,
                    4,
                ),
                (
                    "compute shared memory bytes",
                    glow::MAX_COMPUTE_SHARED_MEMORY_SIZE,
                    32768,
                ),
                (
                    "compute work group invocations",
                    glow::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                    128,
                ),
            ] {
                require_limit(
                    name,
                    i64::from(self.gl.get_parameter_i32(parameter)),
                    required,
                )?;
            }
            // The fixed model's packed weights and each state/scratch arena fit
            // within 16 MiB. Upload code additionally checks the actual sizes
            // against this driver limit before allocating buffers.
            require_limit(
                "SSBO block bytes",
                self.gl
                    .get_parameter_i64(glow::MAX_SHADER_STORAGE_BLOCK_SIZE),
                16 * 1024 * 1024,
            )?;
            for (axis, required) in [(0, 128), (1, 1), (2, 1)] {
                require_limit(
                    "compute local work group size",
                    i64::from(
                        self.gl
                            .get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_SIZE, axis),
                    ),
                    required,
                )?;
                require_limit(
                    "compute work group count",
                    i64::from(
                        self.gl
                            .get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, axis),
                    ),
                    if axis == 0 { 65535 } else { 1 },
                )?;
            }
            let error = self.gl.get_error();
            if error != glow::NO_ERROR {
                return Err(crate::backend(format!(
                    "查询 OpenGL Compute 限制失败：0x{error:04x}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn require_limit(name: &str, actual: i64, required: i64) -> Result<()> {
    if actual < required {
        Err(crate::backend(format!(
            "OpenGL Compute 限制 {name} 为 {actual}，至少需要 {required}"
        )))
    } else {
        Ok(())
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod native {
    use std::ffi::{CStr, c_void};
    use std::num::NonZeroU32;

    use glutin::config::{Api, ConfigSurfaceTypes, ConfigTemplateBuilder};
    use glutin::context::{
        ContextApi, ContextAttributesBuilder, GlProfile, PossiblyCurrentContext, Version,
    };
    use glutin::display::{Display, GlDisplay};
    use glutin::surface::{Surface, SurfaceAttributesBuilder};

    use super::*;

    #[cfg(target_os = "windows")]
    type NativeSurface = glutin::surface::WindowSurface;
    #[cfg(target_os = "linux")]
    type NativeSurface = glutin::surface::PbufferSurface;

    pub(super) struct NativeContext {
        // Unbind in Drop, then Rust drops these in dependency order.
        pub(super) context: PossiblyCurrentContext,
        _surface: Surface<NativeSurface>,
        display: Display,
        #[cfg(target_os = "windows")]
        _window: HiddenWindow,
    }

    impl NativeContext {
        pub(super) fn proc_address(&self, symbol: &CStr) -> *const c_void {
            let address = self.display.get_proc_address(symbol);
            // Some WGL ICDs use small integers or -1 for unsupported symbols.
            match address as usize {
                0 | 1 | 2 | 3 | usize::MAX => std::ptr::null(),
                _ => address,
            }
        }

        #[cfg(target_os = "windows")]
        pub(super) fn new() -> Result<Self> {
            use glutin::display::DisplayApiPreference;
            use raw_window_handle::{RawDisplayHandle, WindowsDisplayHandle};

            let window = HiddenWindow::new()?;
            let handle = window.raw_handle();
            // SAFETY: the real HWND and its class are owned by window, created
            // on this thread and retained until all glutin objects are gone.
            let display = unsafe {
                Display::new(
                    RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                    DisplayApiPreference::Wgl(Some(handle)),
                )
            }
            .map_err(|error| crate::backend(format!("无法创建 WGL display：{error}")))?;
            let template = ConfigTemplateBuilder::new()
                .with_api(Api::OPENGL)
                .with_surface_type(ConfigSurfaceTypes::WINDOW)
                .with_alpha_size(0)
                .prefer_hardware_accelerated(Some(true))
                .compatible_with_native_window(handle)
                .build();
            let config = unsafe { display.find_configs(template) }
                .map_err(|error| crate::backend(format!("无法选择 WGL 硬件像素格式：{error}")))?
                .next()
                .ok_or_else(|| crate::backend("WGL 没有硬件 OpenGL 窗口配置"))?;
            let size = NonZeroU32::new(1).expect("nonzero surface size");
            let attributes =
                SurfaceAttributesBuilder::<NativeSurface>::new().build(handle, size, size);
            let surface = unsafe { display.create_window_surface(&config, &attributes) }
                .map_err(|error| crate::backend(format!("无法创建隐藏 WGL surface：{error}")))?;
            let attributes = ContextAttributesBuilder::new()
                .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 3))))
                .with_profile(GlProfile::Core)
                .build(Some(handle));
            let context = unsafe { display.create_context(&config, &attributes) }
                .map_err(|error| {
                    crate::backend(format!("无法创建 OpenGL 4.3 Core WGL 上下文：{error}"))
                })?
                .make_current(&surface)
                .map_err(|error| crate::backend(format!("无法激活 WGL 上下文：{error}")))?;
            Ok(Self {
                context,
                _surface: surface,
                display,
                _window: window,
            })
        }

        #[cfg(target_os = "linux")]
        pub(super) fn new() -> Result<Self> {
            use glutin::api::egl::{device::Device, display::Display as EglDisplay};

            let devices = Device::query_devices().map_err(|error| {
                crate::backend(format!(
                    "无法枚举无窗口 EGL 设备：{error}；需要支持 EGL_EXT_device_enumeration 和 EGL_EXT_platform_device 的硬件 EGL 驱动，不使用 X11 DISPLAY 或 CPU 回退"
                ))
            })?;
            let mut failures = Vec::new();
            for device in devices {
                let name = device.name().unwrap_or("未命名 EGL 设备");
                if device.extensions().contains("EGL_MESA_device_software") {
                    failures.push(format!("{name}：不支持软件 EGL 设备"));
                    continue;
                }
                // SAFETY: glutin enumerated the device and no native display
                // handle is borrowed. The display is retained by every object.
                let display = match unsafe { EglDisplay::with_device(&device, None) } {
                    Ok(display) => Display::Egl(display),
                    Err(error) => {
                        failures.push(format!("{name}：无法创建 EGL device display：{error}"));
                        continue;
                    }
                };
                match Self::from_egl_display(display) {
                    Ok(context) => return Ok(context),
                    Err(error) => failures.push(format!("{name}: {error}")),
                }
            }
            if failures.is_empty() {
                failures.push("EGL 没有报告任何设备".into());
            }
            Err(crate::backend(format!(
                "没有可用的硬件 OpenGL 4.3 Core EGL pbuffer 上下文：{}。需要配置系统 GPU 的 EGL 驱动；不需要 DISPLAY，也不会回退到 CPU 推理",
                failures.join("; "),
            )))
        }

        #[cfg(target_os = "linux")]
        fn from_egl_display(display: Display) -> Result<Self> {
            let template = ConfigTemplateBuilder::new()
                .with_api(Api::OPENGL)
                .with_surface_type(ConfigSurfaceTypes::PBUFFER)
                .with_alpha_size(0)
                .prefer_hardware_accelerated(Some(true))
                .build();
            // SAFETY: the template contains no borrowed native window.
            let config = unsafe { display.find_configs(template) }
                .map_err(|error| crate::backend(format!("无法选择 EGL pbuffer 配置：{error}")))?
                .next()
                .ok_or_else(|| crate::backend("EGL 设备没有硬件桌面 OpenGL pbuffer 配置"))?;
            let size = NonZeroU32::new(1).expect("nonzero surface size");
            let attributes = SurfaceAttributesBuilder::<NativeSurface>::new().build(size, size);
            // SAFETY: config belongs to display; no native handles are borrowed.
            let surface = unsafe { display.create_pbuffer_surface(&config, &attributes) }.map_err(
                |error| crate::backend(format!("无法创建 EGL pbuffer surface：{error}")),
            )?;
            let attributes = ContextAttributesBuilder::new()
                .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 3))))
                .with_profile(GlProfile::Core)
                .build(None);
            let context = unsafe { display.create_context(&config, &attributes) }
                .map_err(|error| {
                    crate::backend(format!("无法创建 OpenGL 4.3 Core EGL 上下文：{error}"))
                })?
                .make_current(&surface)
                .map_err(|error| crate::backend(format!("无法激活 EGL 上下文：{error}")))?;
            Ok(Self {
                context,
                _surface: surface,
                display,
            })
        }
    }

    impl Drop for NativeContext {
        fn drop(&mut self) {
            // There is no context migration API on our owner. All successful
            // and error-path drops run on the creating worker thread. Do not
            // terminate the process-global EGL display: glutin shares it with
            // other users and manages reference-tracked displays itself.
            let _ = self.context.make_not_current_in_place();
        }
    }

    #[cfg(target_os = "windows")]
    struct HiddenWindow {
        hwnd: windows_sys::Win32::Foundation::HWND,
        instance: windows_sys::Win32::Foundation::HINSTANCE,
        class: Vec<u16>,
    }

    #[cfg(target_os = "windows")]
    impl HiddenWindow {
        fn new() -> Result<Self> {
            use std::sync::atomic::{AtomicU64, Ordering};
            use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
            use windows_sys::Win32::UI::WindowsAndMessaging::{
                CS_OWNDC, CreateWindowExW, DefWindowProcW, RegisterClassW, WNDCLASSW,
                WS_CLIPCHILDREN, WS_CLIPSIBLINGS, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_POPUP,
            };

            static NEXT_CLASS: AtomicU64 = AtomicU64::new(0);
            let class: Vec<u16> = format!(
                "anr-gl-compute-{}-{}",
                std::process::id(),
                NEXT_CLASS.fetch_add(1, Ordering::Relaxed),
            )
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
            // SAFETY: null requests the calling executable's module, which
            // remains loaded throughout the worker's lifetime.
            let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
            if instance.is_null() {
                return Err(crate::backend(format!(
                    "无法获取隐藏 OpenGL 窗口的模块：{}",
                    std::io::Error::last_os_error(),
                )));
            }
            let window_class = WNDCLASSW {
                style: CS_OWNDC,
                lpfnWndProc: Some(DefWindowProcW),
                hInstance: instance,
                lpszClassName: class.as_ptr(),
                // SAFETY: zero is valid for the unused optional handles,
                // extra-byte counts and menu name of this window class.
                ..unsafe { std::mem::zeroed() }
            };
            if unsafe { RegisterClassW(&window_class) } == 0 {
                return Err(crate::backend(format!(
                    "无法注册隐藏 OpenGL 窗口类：{}",
                    std::io::Error::last_os_error(),
                )));
            }
            // Establish the class guard before the next fallible operation.
            let mut window = Self {
                hwnd: std::ptr::null_mut(),
                instance,
                class,
            };
            // SAFETY: class was registered above. No WS_VISIBLE, ShowWindow,
            // parent, menu, or event loop: this is a real but never shown HWND,
            // not a message-only window (which is unsuitable for WGL).
            window.hwnd = unsafe {
                CreateWindowExW(
                    WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                    window.class.as_ptr(),
                    window.class.as_ptr(),
                    WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
                    0,
                    0,
                    1,
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    instance,
                    std::ptr::null(),
                )
            };
            if window.hwnd.is_null() {
                return Err(crate::backend(format!(
                    "无法创建隐藏 OpenGL 窗口：{}",
                    std::io::Error::last_os_error(),
                )));
            }
            Ok(window)
        }

        fn raw_handle(&self) -> raw_window_handle::RawWindowHandle {
            use std::num::NonZeroIsize;
            let mut handle = raw_window_handle::Win32WindowHandle::new(
                NonZeroIsize::new(self.hwnd as isize).expect("owned window is non-null"),
            );
            handle.hinstance = NonZeroIsize::new(self.instance as isize);
            raw_window_handle::RawWindowHandle::Win32(handle)
        }
    }

    #[cfg(target_os = "windows")]
    impl Drop for HiddenWindow {
        fn drop(&mut self) {
            use windows_sys::Win32::UI::WindowsAndMessaging::{DestroyWindow, UnregisterClassW};
            // SAFETY: this worker owns the class and window, and glutin's
            // context, surface/DC and display have already been destroyed.
            unsafe {
                if !self.hwnd.is_null() {
                    DestroyWindow(self.hwnd);
                }
                UnregisterClassW(self.class.as_ptr(), self.instance);
            }
        }
    }
}
