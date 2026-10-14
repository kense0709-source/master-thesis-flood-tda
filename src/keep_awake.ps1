# 長時間計算の間、モダンスタンバイと画面消灯を抑える（progress.md 2026-10-05 の対策）
#
# 画面が消えると AC 接続でもスタンバイに入り、計算がほぼ止まる（10-05〜06 に約7時間停止）。
# 指定したプロセスが終わるまで 30 秒ごとに SetThreadExecutionState を呼び、終了時に解除する。
# ふたを閉じると止まる点は防げない。
#
# 使い方（計算プロセスの PID を渡す。独立プロセスとして起動すれば VSCode を閉じても続く）:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'src\keep_awake.ps1', '-TargetPid', 12345
param([Parameter(Mandatory = $true)][int]$TargetPid)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
$flags = [uint32]'0x80000003'
while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
    [Win32.Power]::SetThreadExecutionState($flags) | Out-Null
    Start-Sleep -Seconds 30
}
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
