#define MyAppName "Tram Can QR"
#define MyAppVersion "0.2.0-rc8"
#define MyAppExeName "TramCanQR.exe"

[Setup]
AppId={{78F30B4A-47A5-4B9C-A183-E6B7E5E5A241}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Viet Nhat IPT
DefaultDirName={localappdata}\Programs\TramCanQR
DefaultGroupName={#MyAppName}
OutputDir=..\dist\installer
OutputBaseFilename=TramCanQR-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\TramCanQR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\packaging\customer-config.env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\packaging\gemini-pilot-config.env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\packaging\HUONG-DAN-KHACH-HANG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docs\GEMINI-COST.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\packaging\gemini-pilot-config.env.example"; DestDir: "{localappdata}\TramCanQR"; DestName: "config.env"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Tạo biểu tượng ngoài màn hình"; GroupDescription: "Biểu tượng:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Mở {#MyAppName}"; Flags: nowait postinstall skipifsilent
