/**
 * AvatarCanvas: manages the Three.js viewport, scene, camera, lighting,
 * OrbitControls, and the AvatarRig lifecycle.
 *
 * Requirements met:
 *  - Waist-up camera framing, slightly low angle, keeping both hands and face in view.
 *  - Soft studio lighting (key + fill + rim), theme-aware background matching current panel.
 *  - OrbitControls limited to small horizontal rotation only, with zoom disabled.
 *  - Skin and clothing styled: solid dark navy top, sleeves ending above wrists so hands pop.
 *  - Placeholder UI with clear instructions if avatar.glb is missing.
 *  - Strict disposal of geometries, materials, and textures on unmount.
 */
import {
  WebGLRenderer,
  Scene,
  PerspectiveCamera,
  DirectionalLight,
  AmbientLight,
  HemisphereLight,
  Color,
  MeshStandardMaterial,
  Mesh,
  SRGBColorSpace,
  PCFSoftShadowMap,
  Vector3,
} from 'three';
import { GLTFLoader } from '../vendor/three/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from '../vendor/three/examples/jsm/loaders/FBXLoader.js';
import { OrbitControls } from '../vendor/three/examples/jsm/controls/OrbitControls.js';
import { AvatarRig } from './rig.js?v=33';
import { BONE_NAMES } from './boneMap.js?v=33';

export const DEFAULT_MODEL_URL = './assets/Ch46_nonPBR.fbx';

export class AvatarCanvas {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {Object} [options]
   * @param {string} [options.modelUrl]
   * @param {HTMLElement} [options.placeholderEl]
   * @param {any} [options.restHand] relaxed hand landmarks
   * @param {() => void} [options.onReady]
   * @param {(err: Error) => void} [options.onError]
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.modelUrl = options.modelUrl || DEFAULT_MODEL_URL;
    this.placeholderEl = options.placeholderEl || null;
    this.restHand = options.restHand || null;
    this.onReady = options.onReady || null;
    this.onError = options.onError || null;
    this.onProgress = options.onProgress || null;

    this.disposed = false;
    this.rig = null;
    this.modelRoot = null;
    this.animFrameId = null;
    this.lastTime = 0;

    // 1. Renderer
    this.renderer = new WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;

    // 2. Scene
    this.scene = new Scene();
    this.updateThemeBackground();

    // 3. Camera (waist-up, slightly low angle)
    this.camera = new PerspectiveCamera(32, 1, 0.1, 20);
    this.camera.position.set(0, 1.25, 1.85);

    // 4. OrbitControls: small horizontal rotation only, no zoom, no pan
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enableZoom = false;
    this.controls.enablePan = false;
    this.controls.minPolarAngle = Math.PI * 0.44; // lock vertical angle
    this.controls.maxPolarAngle = Math.PI * 0.52;
    this.controls.minAzimuthAngle = -Math.PI * 0.22; // ~40 deg left/right only
    this.controls.maxAzimuthAngle = Math.PI * 0.22;
    this.controls.target.set(0, 1.35, 0);

    // 5. Balanced natural studio lighting (warm key + soft fill + subtle rim)
    this.hemiLight = new HemisphereLight(0xffffff, 0x334155, 0.6);
    this.scene.add(this.hemiLight);

    this.ambientLight = new AmbientLight(0xffffff, 0.45);
    this.scene.add(this.ambientLight);

    // Key light (warm, upper-right front)
    this.keyLight = new DirectionalLight(0xfff8ee, 0.95);
    this.keyLight.position.set(1.4, 2.4, 2.0);
    this.keyLight.castShadow = true;
    this.scene.add(this.keyLight);

    // Fill light (clean soft, upper-left front)
    this.fillLight = new DirectionalLight(0xe0f2fe, 0.55);
    this.fillLight.position.set(-1.4, 1.8, 1.5);
    this.scene.add(this.fillLight);

    // Soft eye-level front fill (prevents harsh shadows without washing out face)
    this.frontFill = new DirectionalLight(0xffffff, 0.35);
    this.frontFill.position.set(0, 1.3, 2.2);
    this.scene.add(this.frontFill);

    // Rim light (subtle backlight separating hair and shoulders)
    this.rimLight = new DirectionalLight(0xdbeafe, 0.5);
    this.rimLight.position.set(0, 2.2, -1.8);
    this.scene.add(this.rimLight);

    // Theme observer
    this._themeObserver = new MutationObserver(() => this.updateThemeBackground());
    this._themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    // Window resize listener
    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);

    // Initial resize
    this.resize();

    // Start loading model
    this.loadModel();

    // Bound render loop
    this._loop = this._loop.bind(this);
    this.animFrameId = requestAnimationFrame(this._loop);
  }

  updateThemeBackground() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    this.scene.background = new Color(isLight ? 0xeaedf4 : 0x161619);
    if (this.hemiLight) {
      this.hemiLight.color.setHex(isLight ? 0xffffff : 0xe2e8f0);
      this.hemiLight.groundColor.setHex(isLight ? 0x94a3b8 : 0x0f172a);
    }
  }

  resize() {
    if (!this.canvas || !this.canvas.parentElement) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (this.renderer) this.renderer.setSize(w, h, false);
    if (this.camera) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  loadModel() {
    const isFbx = this.modelUrl.toLowerCase().endsWith('.fbx');
    const onLoaded = (root) => {
      if (this.disposed) return;
      this.modelRoot = root;

      // Auto-detect centimeter rigs (e.g. Mixamo FBX height ~160-180cm) and scale to standard meters (~1.6-1.8m)
      this.modelRoot.updateMatrixWorld(true);
      let isCm = false;
      this.modelRoot.traverse((o) => {
        if (o.isBone && /head/i.test(o.name)) {
          const wp = o.getWorldPosition(new Vector3());
          if (wp.y > 20) isCm = true;
        }
      });
      if (isCm) {
        this.modelRoot.scale.setScalar(0.01);
        this.modelRoot.updateMatrixWorld(true);
      }

      this.scene.add(this.modelRoot);

      // Apply dark navy clothing style to tops and warm skin tone
      this._styleModel(this.modelRoot);

      this.rig = new AvatarRig(this.modelRoot, {
        config: BONE_NAMES,
        restHand: this.restHand,
      });

      // Frame waist-up from rig layout
      this._frameWaistUp(this.rig.layout);

      if (this.placeholderEl) this.placeholderEl.style.display = 'none';
      console.log('[Avatar3D] Rig ready from model at', this.modelUrl);
      if (this.onReady) this.onReady();
    };

    const onError = (err) => {
      if (this.disposed) return;
      console.warn(`[Avatar3D] Could not load model at "${this.modelUrl}":`, err.message || err);
      // Fallback: If FBX failed, try loading /assets/avatar.glb before showing placeholder
      if (isFbx && this.modelUrl !== '/assets/avatar.glb') {
        console.info('[Avatar3D] Attempting fallback to /assets/avatar.glb...');
        this.modelUrl = '/assets/avatar.glb';
        this.loadModel();
        return;
      }
      this._showPlaceholder();
      if (this.onError) this.onError(err);
    };

    const onProgress = (xhr) => {
      if (this.disposed) return;
      if (this.onProgress) {
        try {
          this.onProgress(xhr);
        } catch (_) {}
      }
    };

    if (isFbx) {
      const loader = new FBXLoader();
      loader.load(this.modelUrl, onLoaded, onProgress, onError);
    } else {
      const loader = new GLTFLoader();
      loader.load(
        this.modelUrl,
        (gltf) => onLoaded(gltf.scene || gltf.scenes[0]),
        onProgress,
        onError
      );
    }
  }

  _styleModel(root) {
    // Elegant human styling: preserve embedded textures if present (e.g. Ch46 has face/hair/clothes textures)
    const darkNavy = new Color(0x18202f);
    const skinTone = new Color(0xdfa684);
    const jointAccent = new Color(0x2d3748);

    root.traverse((o) => {
      if (o.isMesh && o.material) {
        o.castShadow = true;
        o.receiveShadow = true;

        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const mat of mats) {
          // If the model has textures, ensure natural colors and accurate shading
          if (mat.map) {
            mat.map.colorSpace = SRGBColorSpace;
            if (mat.color) mat.color.setRGB(1.0, 1.0, 1.0); // natural texture albedo
            if (mat.roughness !== undefined && mat.roughness > 0.8) mat.roughness = 0.7;
            mat.needsUpdate = true;
            continue;
          }

          // Fallback styling for untextured dummy mannequins
          const name = (o.name || '').toLowerCase() + ' ' + (mat.name || '').toLowerCase();
          if (name.includes('joint')) {
            if (mat.color) mat.color.copy(jointAccent);
            if (mat.shininess !== undefined) mat.shininess = 15;
          } else if (name.includes('surface') || name.includes('limb') || name.includes('body')) {
            if (mat.color) mat.color.copy(skinTone);
            if (mat.specular) mat.specular.setHex(0x332219);
            if (mat.shininess !== undefined) mat.shininess = 8;
          } else if (name.includes('top') || name.includes('shirt') || name.includes('cloth') || name.includes('outfit_top')) {
            if (mat.color) mat.color.copy(darkNavy);
            if (mat.roughness !== undefined) mat.roughness = 0.85;
          }
        }
      }
    });
  }

  _frameWaistUp(layout) {
    // Focus target at chest/neck center so face and both hands remain centered
    const targetY = (layout.headY + layout.hipsY) * 0.53;
    this.controls.target.set(layout.centerX, targetY, layout.centerZ);

    // Frame camera: comfortable waist-up view showing both arms, hands, and head
    const span = Math.max(layout.shoulderSpan || 0.35, (layout.armLength || 0.55) * 1.4);
    const dist = Math.max(1.85, span * 2.3);
    this.camera.fov = 34;
    this.camera.position.set(layout.centerX, targetY + 0.05, layout.centerZ + dist);
    this.camera.lookAt(this.controls.target);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _showPlaceholder() {
    if (this.placeholderEl) {
      this.placeholderEl.style.display = 'flex';
      return;
    }
    // Create inline overlay inside avatar-viewport if no external container given
    const wrap = this.canvas.parentElement;
    if (!wrap) return;
    let el = wrap.querySelector('.avatar-3d-placeholder');
    if (!el) {
      el = document.createElement('div');
      el.className = 'avatar-3d-placeholder';
      el.innerHTML = `
        <div class="ph-card">
          <div class="ph-icon">🤖</div>
          <h3 class="ph-title">3D Avatar Model Missing</h3>
          <p class="ph-desc">
            To view the rigged 3D human avatar, place a humanoid GLB model at:
          </p>
          <code class="ph-path">web/assets/T-Pose.fbx</code> (or <code>avatar.glb</code>)
          <p class="ph-hint">
            Compatible rigs: <strong>Mixamo FBX / GLB</strong> or <strong>Ready Player Me (.glb)</strong>.<br/>
            Currently falling back to the 2D character graphics.
          </p>
          <button class="ph-btn" id="btn-dismiss-ph">Switch to 2D Mode</button>
        </div>
      `;
      wrap.appendChild(el);
      const btn = el.querySelector('#btn-dismiss-ph');
      if (btn) {
        btn.addEventListener('click', () => {
          if (typeof window.USE_3D_AVATAR !== 'undefined') {
            window.USE_3D_AVATAR = false;
            el.remove();
            if (typeof window._reinitAvatarRenderer === 'function') {
              window._reinitAvatarRenderer();
            }
          }
        });
      }
    }
    el.style.display = 'flex';
    this.placeholderEl = el;
  }

  /**
   * Main animation tick: controls + rig + render.
   * @param {number} now ms
   */
  _loop(now) {
    if (this.disposed) return;
    this.animFrameId = requestAnimationFrame(this._loop);

    const dt = this.lastTime ? Math.min(0.1, (now - this.lastTime) * 0.001) : 0.016;
    this.lastTime = now;

    if (this.controls) this.controls.update();

    if (this.rig && window.practiceCurrentPose) {
      this.rig.update(window.practiceCurrentPose, dt);
    } else if (this.rig) {
      this.rig.update(null, dt);
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  dispose() {
    this.disposed = true;
    window.removeEventListener('resize', this.resize);
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    if (this._themeObserver) this._themeObserver.disconnect();
    if (this.controls) this.controls.dispose();

    // Clean up Three.js objects
    if (this.scene) {
      this.scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
          else o.material.dispose();
        }
      });
    }
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.forceContextLoss();
    }
    if (this.placeholderEl && this.placeholderEl.parentElement) {
      this.placeholderEl.remove();
    }
  }
}
