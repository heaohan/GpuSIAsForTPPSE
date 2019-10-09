for($i=1;$i -le 20;$i++) {
  mkdir ./$i
  
  <#
  if ($i -eq 5 -or $i -eq 10 -or $i -eq 15 -or $i -eq 20) {
    [System.Media.SystemSounds]::Hand.Play();
    pause;
  }
  #>
  
  <#
  APSO.exe
  DE.exe
  Firefly.exe
  PSO.exe
  GPU_APSO.exe
  GPU_DE.exe
  GPU_Firefly.exe
  GPU_PSO.exe
  PSO_modified.exe
  APSO_modified.exe
  #>
  .\APSO_modified.exe
  
  Copy-Item ".\*.mat" -Exclude "for_c.mat" -Destination ".\$i"
}
[System.Media.SystemSounds]::Hand.Play();