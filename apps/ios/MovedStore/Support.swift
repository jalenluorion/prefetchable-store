import Foundation
import Darwin
import os

// Small shared helpers: the console log, the Documents folder, the process's memory reading for the status dots and
// Diagnostics, names for the thermal state and the device.

func log(_ s: String) {
    print(s)
    fflush(stdout)
}

let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
/// The process start (MemSample.t).
let appStart = Date()

/// One memory reading. Bytes; GB in the console line = 1e9 B.
struct MemSample: Codable, Sendable {
    var label: String
    var t: Double                     // seconds since launch
    /// task_vm_info.phys_footprint: what iOS charges the app against its memory limit.
    var physFootprint: Int
    /// task_vm_info.ledger_phys_footprint_peak: the footprint's high-water mark since launch.
    var physFootprintPeak: Int
    var residentSize: Int             // task_vm_info.resident_size (includes clean file-backed pages)
    var compressed: Int               // task_vm_info.compressed
    /// os_proc_available_memory(): how much more the app may take before iOS ends it (0 on the Mac).
    var osProcAvailable: Int
    /// physFootprint + osProcAvailable: the app's memory limit as iOS sees it now (nil on the Mac).
    var limitEstimate: Int?
    var thermal: String

    var line: String {
        func g(_ b: Int) -> String { String(format: "%.3f", Double(b) / 1e9) }
        return "MEM \(label): footprint=\(g(physFootprint)) peak=\(g(physFootprintPeak)) avail=\(g(osProcAvailable))"
            + " resident=\(g(residentSize)) compressed=\(g(compressed)) GB thermal=\(thermal)"
    }
}

/// The process's memory. Cheap (one task_info call); any thread.
enum MemProbe {
    static func vmInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info : nil
    }

    static func read(_ label: String) -> MemSample {
        let vm = vmInfo()
        let avail = Int(os_proc_available_memory())
        let fp = Int(vm?.phys_footprint ?? 0)
        return MemSample(label: label, t: Date().timeIntervalSince(appStart), physFootprint: fp,
                         physFootprintPeak: Int(vm?.ledger_phys_footprint_peak ?? 0),
                         residentSize: Int(vm?.resident_size ?? 0), compressed: Int(vm?.compressed ?? 0),
                         osProcAvailable: avail, limitEstimate: avail > 0 ? fp + avail : nil,
                         thermal: thermalName(ProcessInfo.processInfo.thermalState))
    }
}

func thermalName(_ s: ProcessInfo.ThermalState) -> String {
    switch s {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func machineName() -> String {
    var u = utsname()
    uname(&u)
    return withUnsafeBytes(of: &u.machine) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
