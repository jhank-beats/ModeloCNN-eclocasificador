# Guía Rápida para Compilar el APK (.apk)

Para compilar el archivo `.apk` instalable en cualquier teléfono Android, dispones de dos alternativas sencillas y directas:

---

### Opción 1: Compilar el APK con Flutter (Recomendado - 2 Comandos)
Como el proyecto Flutter ya está completamente estructurado con sus assets, código y dependencias en `momento tres/semana 13/flutter_app_residuos/`:

1. Abre tu terminal o máquina donde tengas Flutter y Android Studio instalados.
2. Navega al proyecto:
   ```bash
   cd "momento tres/semana 13/flutter_app_residuos"
   ```
3. Obtén los paquetes:
   ```bash
   flutter pub get
   ```
4. Compila el APK optimizado para release:
   ```bash
   flutter build apk --release
   ```
5. El archivo `.apk` final quedará generado en:
   `build/app/outputs/flutter-apk/app-release.apk`
   *(Pesa aproximadamente 18-22 MB e incluye el motor TFLite embebido listo para instalar en cualquier teléfono).*

---

### Opción 2: Compilación Gratuita en la Nube con GitHub Actions (Sin Instalar Nada Local)
Si no tienes Flutter o Android Studio en este equipo, puedes compilar el APK en 3 minutos usando GitHub Actions:

Crea un archivo `.github/workflows/build_apk.yml` con este contenido:
```yaml
name: Build APK Bienal UCC

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v3
        with:
          distribution: 'zulu'
          java-version: '17'
      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.19.x'
          channel: 'stable'
      - name: Build APK
        run: |
          cd "momento tres/semana 13/flutter_app_residuos"
          flutter pub get
          flutter build apk --release
      - uses: actions/upload-artifact@v4
        with:
          name: EcoClasificador-Bienal-UCC
          path: momento tres/semana 13/flutter_app_residuos/build/app/outputs/flutter-apk/app-release.apk
```
Al hacer `git push`, GitHub compila el APK y te da el enlace directo para descargarlo e instalarlo en tu teléfono.

---

### Opción 3: Demostración Inmediata en el Teléfono (PWA Móvil)
Si necesitas mostrar la app **ya mismo en tu teléfono** sin esperar a compilar el APK:
1. En esta máquina corre:
   ```bash
   cd "momento tres/semana 13/web_mobile_pwa" && python -m http.server 8080
   ```
2. Desde tu teléfono conectado a la misma red Wi-Fi abre:
   `http://<IP-DE-TU-PC>:8080`
3. En Chrome de tu teléfono presiona **"Agregar a la pantalla de inicio"** y se instalará como una app nativa con acceso directo, cámara y funcionamiento fluido.
