; Windows installer for the DeepReefMap PyApp binary.
;
; The installer runs once: it places the binary at a fixed per-user path and
; creates shortcuts pointing there. In-app updates and rollbacks swap the
; binary at that same path (packaging/binary_swap.py), so shortcuts never break and
; the installer is never re-run. bootstrap.py refreshes the DisplayVersion
; registry value on launch so Add/Remove Programs tracks in-app updates.
;
; Build (CI passes all three defines):
;   iscc /DBinaryPath=dist\deepreefmap-windows-x64-1.2.0.exe ^
;        /DAppVersion=1.2.0 /DOutputName=deepreefmap-setup-windows-x64-1.2.0 ^
;        scripts\installer.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef OutputName
  #define OutputName "deepreefmap-setup-windows-x64"
#endif

[Setup]
AppId=DeepReefMap
; Relative paths below (BinaryPath, icon, OutputDir) resolve from the repo
; root, not scripts/.
SourceDir=..
AppName=DeepReefMap
AppVersion={#AppVersion}
AppPublisher=EPFL ECEO
AppPublisherURL=https://github.com/eceo-epfl/deepreefmap
DefaultDirName={userpf}\DeepReefMap
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename={#OutputName}
SetupIconFile=dist\icon.ico
UninstallDisplayIcon={app}\icon.ico
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked

[Files]
Source: "{#BinaryPath}"; DestDir: "{app}"; DestName: "deepreefmap.exe"; Flags: ignoreversion
Source: "dist\icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\DeepReefMap"; Filename: "{app}\deepreefmap.exe"; IconFilename: "{app}\icon.ico"
Name: "{userdesktop}\DeepReefMap"; Filename: "{app}\deepreefmap.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
; Provision the Python environment during install so the first launch is
; instant and console-free. `self restore` is PyApp-internal and installs the
; env fresh. Inno ignores the exit code, so an offline install still completes
; and the first launch provisions silently instead (hidden console).
Filename: "{app}\deepreefmap.exe"; Parameters: "self restore"; StatusMsg: "Setting up Python environment (this may take several minutes)..."; Flags: runhidden waituntilterminated
Filename: "{app}\deepreefmap.exe"; Description: "Launch DeepReefMap"; Flags: nowait postinstall skipifsilent

[Code]
// Uninstall removes the app and its provisioned Python environment. User data
// (Documents\DeepReefMap outputs, settings) is never touched; downloaded model
// weights are multi-GB but re-downloadable, so their removal is opt-in.

procedure DeleteHfCacheModels(const Prefix: String);
var
  HubDir: String;
  FindRec: TFindRec;
begin
  HubDir := ExpandConstant('{%USERPROFILE}') + '\.cache\huggingface\hub\';
  if FindFirst(HubDir + Prefix, FindRec) then begin
    try
      repeat
        if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          DelTree(HubDir + FindRec.Name, True, True, True);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then begin
    // The PyApp-provisioned Python environment for this app (all versions).
    DelTree(ExpandConstant('{localappdata}\pyapp\data\deepreefmap'), True, True, True);

    if MsgBox('Also remove downloaded AI models (several GB, re-downloadable)?'
              + #13#10 + 'Reconstruction outputs in Documents\DeepReefMap are kept either way.',
              mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then begin
      // LoGeR checkpoints + app data (platformdirs user_data_dir("deepreefmap")).
      DelTree(ExpandConstant('{localappdata}\deepreefmap'), True, True, True);
      // Hugging Face cache entries for the model repos the app downloads.
      DeleteHfCacheModels('models--EPFL-ECEO--*');
      DeleteHfCacheModels('models--facebook--dinov3*');
      DeleteHfCacheModels('models--nvidia--segformer*');
      DeleteHfCacheModels('models--Junyi42--*');
    end;
  end;
end;
