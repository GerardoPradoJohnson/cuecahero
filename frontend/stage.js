import * as T from './vendor/three.module.js';

// Presentation only: pixels/actions/events from the same live or replay snapshot.
export class FlyStage {
  constructor(canvas) {
    this.lastRender=-Infinity; this.lastImage=null; this.state=null; this.eventUntil=0; this.event=null;
    this.reduced=matchMedia('(prefers-reduced-motion: reduce)');
    this.renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'low-power'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
    this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=T.PCFSoftShadowMap;
    this.renderer.outputColorSpace=T.SRGBColorSpace;this.renderer.toneMapping=T.ACESFilmicToneMapping;
    this.scene=new T.Scene();this.scene.background=new T.Color('#272532');
    this.camera=new T.PerspectiveCamera(36,1,.1,60);
    this.view=0; this.angle=0; this.zoom=1;
    this.scene.add(new T.HemisphereLight('#e8dcff','#54403f',2.5));
    const key=new T.DirectionalLight('#ffe4bb',4);key.position.set(-3,7,5);key.castShadow=true;
    key.shadow.mapSize.set(1024,1024);Object.assign(key.shadow.camera,{left:-6,right:6,top:6,bottom:-6});key.shadow.bias=-.001;this.scene.add(key);
    const rim=new T.DirectionalLight('#a998ff',2.5);rim.position.set(4,4,-4);this.scene.add(rim);
    this.screenLight=new T.PointLight('#b7a0ff',8,7,2);this.screenLight.position.set(1,2,0);this.scene.add(this.screenLight);
    this.buildRoom();this.buildTV();this.buildFly();
    let drag=null;
    canvas.addEventListener('pointerdown',e=>{drag=e.clientX;canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{if(drag===null)return;this.angle=T.MathUtils.clamp(this.angle+(e.clientX-drag)*.003,-.32,.32);drag=e.clientX;});
    for(const name of ['pointerup','pointercancel'])canvas.addEventListener(name,()=>drag=null);
    canvas.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();this.angle=T.MathUtils.clamp(this.angle+(e.key==='ArrowLeft'?-.08:.08),-.32,.32);}});
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();document.querySelector('.game-wrap').classList.add('stage-failed');});
    canvas.addEventListener('webglcontextrestored',()=>document.querySelector('.game-wrap').classList.remove('stage-failed'));
  }
  material(color,extra={}){return new T.MeshStandardMaterial({color,roughness:.72,flatShading:true,...extra});}
  mesh(geometry,material,parent=this.scene,pos=[0,0,0]){const m=new T.Mesh(geometry,material);m.position.set(...pos);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
  box(size,color,parent,pos){return this.mesh(new T.BoxGeometry(...size),this.material(color),parent,pos);}
  ellipsoid(size,color,parent,pos,detail=1){const m=this.mesh(new T.IcosahedronGeometry(1,detail),typeof color==='string'?this.material(color):color,parent,pos);m.scale.set(...size);return m;}
  rod(a,b,r,color,parent=this.scene){const p=new T.Vector3(...a),q=new T.Vector3(...b),v=q.clone().sub(p);const m=this.mesh(new T.CylinderGeometry(r*.8,r,v.length(),6),typeof color==='string'?this.material(color):color,parent);m.position.copy(p.add(q).multiplyScalar(.5));m.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),v.normalize());return m;}
  label(text,w,h,color='#d9c7e9',background=null){const c=document.createElement('canvas');c.width=512;c.height=128;const ctx=c.getContext('2d');if(background){ctx.fillStyle=background;ctx.fillRect(0,0,512,128);}ctx.fillStyle=color;ctx.font='500 40px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,256,64);const texture=new T.CanvasTexture(c);texture.colorSpace=T.SRGBColorSpace;return new T.Mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false}));}
  createChileanFlagTexture(){
    const c=document.createElement('canvas');c.width=600;c.height=400;const ctx=c.getContext('2d');
    ctx.fillStyle='#0033a0';ctx.fillRect(0,0,200,200);
    ctx.fillStyle='#ffffff';ctx.beginPath();
    const cx=100,cy=100,outer=46,inner=18;
    for(let i=0;i<5;i++){
      const aOut=-Math.PI/2+(i*2*Math.PI)/5;
      const aIn=aOut+Math.PI/5;
      if(i===0)ctx.moveTo(cx+Math.cos(aOut)*outer,cy+Math.sin(aOut)*outer);
      else ctx.lineTo(cx+Math.cos(aOut)*outer,cy+Math.sin(aOut)*outer);
      ctx.lineTo(cx+Math.cos(aIn)*inner,cy+Math.sin(aIn)*inner);
    }
    ctx.closePath();ctx.fill();
    ctx.fillStyle='#f6f7f9';ctx.fillRect(200,0,400,200);
    ctx.fillStyle='#d52b1e';ctx.fillRect(0,200,600,200);
    const tex=new T.CanvasTexture(c);tex.colorSpace=T.SRGBColorSpace;return tex;
  }
  createFondaSignTexture(){
    const c=document.createElement('canvas');c.width=800;c.height=200;const ctx=c.getContext('2d');
    ctx.fillStyle='#362013';ctx.fillRect(0,0,800,200);
    for(let i=0;i<4;i++){ctx.fillStyle=i%2?'#3c2416':'#301a0f';ctx.fillRect(0,i*50,800,48);}
    ctx.strokeStyle='#d9a557';ctx.lineWidth=6;ctx.strokeRect(14,14,772,172);
    ctx.strokeStyle='#f5dd94';ctx.lineWidth=2;ctx.strokeRect(22,22,756,156);
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.font='bold 44px system-ui, sans-serif';ctx.fillStyle='#fff6de';
    ctx.fillText('FONDA "LA MOSCA CUEQUERA"',400,74);
    ctx.font='bold 26px monospace';ctx.fillStyle='#f0bd70';
    ctx.fillText('★ TIKI TIKI TI · 18 DE SEPTIEMBRE ★',400,132);
    const tex=new T.CanvasTexture(c);tex.colorSpace=T.SRGBColorSpace;return tex;
  }
  buildBunting(p1,p2,sag=0.22,num=10){
    const group=new T.Group();
    const v1=new T.Vector3(...p1),v2=new T.Vector3(...p2);
    const pts=[];
    for(let i=0;i<=16;i++){
      const t=i/16;
      const p=new T.Vector3().lerpVectors(v1,v2,t);
      p.y-=Math.sin(t*Math.PI)*sag;
      pts.push(p);
    }
    const curve=new T.CatmullRomCurve3(pts);
    this.mesh(new T.TubeGeometry(curve,24,.008,4,false),this.material('#e8d5bf'),group);
    const colors=['#d52b1e','#ffffff','#0033a0'];
    const triShape=new T.Shape();
    triShape.moveTo(-.085,0);triShape.lineTo(.085,0);triShape.lineTo(0,-.20);triShape.closePath();
    const triGeom=new T.ShapeGeometry(triShape);
    const angle=(v2.z-v1.z>0.01||v2.z-v1.z<-0.01)?Math.atan2(v2.x-v1.x,v2.z-v1.z):0;
    for(let i=1;i<=num;i++){
      const t=i/(num+1);
      const pos=curve.getPoint(t);
      const flag=this.mesh(triGeom,this.material(colors[(i-1)%3],{side:T.DoubleSide,roughness:.6}),group,[pos.x,pos.y,pos.z]);
      flag.rotation.y=angle;
      if(i%2===1){
        this.ellipsoid([.025,.032,.025],this.material('#fff0aa',{emissive:'#ffcc44',emissiveIntensity:1.8}),group,[pos.x,pos.y+.02,pos.z],0);
      }
    }
    this.scene.add(group);
    return group;
  }
  buildRoom(){
    this.box([8,.25,6.5],'#665a50',this.scene,[0,-.17,0]);
    for(let x=-3.6;x<=3.6;x+=.9)this.box([.02,.02,6.4],'#423832',this.scene,[x,-.04,0]);
    this.box([8,3.9,.14],'#453c39',this.scene,[0,1.7,-3]);
    for(let x=-3.7;x<4;x+=.45)this.box([.035,3.8,.04],'#534743',this.scene,[x,1.7,-2.90]);

    // Ramada wooden structure
    this.rod([-3.7,0,2.4],[-3.7,3.5,2.4],.075,'#54361c');
    this.rod([3.7,0,2.4],[3.7,3.5,2.4],.075,'#54361c');
    this.rod([-3.7,0,-2.8],[-3.7,3.5,-2.8],.075,'#54361c');
    this.rod([3.7,0,-2.8],[3.7,3.5,-2.8],.075,'#54361c');
    this.rod([-3.75,3.45,2.4],[3.75,3.45,2.4],.065,'#4d3119');
    this.rod([-3.75,3.45,-2.8],[3.75,3.45,-2.8],.065,'#4d3119');
    for(let x of [-3.7,-1.8,0,1.8,3.7]){
      this.rod([x,3.5,2.45],[x,3.5,-2.85],.055,'#482d16');
    }
    for(let corner of [[-3.7,3.35,2.3],[3.7,3.35,2.3],[-3.7,3.35,-2.7],[3.7,3.35,-2.7]]){
      for(let j=0;j<4;j++){
        const leaf=this.ellipsoid([.18,.32,.08],j%2?'#3e5e39':'#51734a',this.scene,[corner[0]+(j%2?-.12:.12),corner[1]-(j*.07),corner[2]+(j>1?.1:-.1)],0);
        leaf.rotation.set(.4*j,.5*j,-.3*j);
      }
    }

    // Bandera chilena (ubicada en la pared izquierda, completamente visible sin tapar la TV)
    const flagTex=this.createChileanFlagTexture();
    this.mesh(new T.PlaneGeometry(1.9,1.26),new T.MeshStandardMaterial({map:flagTex,roughness:.65}),this.scene,[-2.05,2.35,-2.86]);
    this.rod([-3.05,3.02,-2.84],[-1.05,3.02,-2.84],.022,'#3d2312');
    this.ellipsoid([.04,.04,.04],this.material('#dfb152',{metalness:.6,roughness:.3}),this.scene,[-3.08,3.02,-2.84],0);
    this.ellipsoid([.04,.04,.04],this.material('#dfb152',{metalness:.6,roughness:.3}),this.scene,[-1.02,3.02,-2.84],0);

    // Warm ambient fonda light near the barrel
    const fondaWarmLight=new T.PointLight('#ffaa44',3.5,4.5,1.8);
    fondaWarmLight.position.set(-2.8,1.5,.6);
    this.scene.add(fondaWarmLight);

    // Cartel de fonda
    const signTex=this.createFondaSignTexture();
    const signMesh=this.mesh(new T.PlaneGeometry(2.2,.55),new T.MeshStandardMaterial({map:signTex,roughness:.6}),this.scene,[1.1,3.22,-1.4]);
    signMesh.rotation.y=-.12;
    this.rod([.35,3.5,-1.31],[.35,3.45,-1.31],.012,'#e0ceb5');
    this.rod([1.85,3.5,-1.49],[1.85,3.45,-1.49],.012,'#e0ceb5');

    // Guirnaldas de banderines y luces
    this.buildBunting([-3.7,3.3,-2.7],[3.7,3.3,-2.7],.28,14);
    this.buildBunting([-3.6,3.35,2.2],[3.6,3.35,2.2],.26,12);
    this.buildBunting([-3.5,3.35,1.9],[3.5,3.3,-2.0],.32,13);

    // Rug
    const rug=this.mesh(new T.CylinderGeometry(2.45,2.45,.025,8),this.material('#a38472'),this.scene,[-.2,-.025,.5]);rug.scale.z=.75;
    for(let i=0;i<3;i++){const ring=this.mesh(new T.TorusGeometry(1.85+i*.14,.012,4,8),this.material('#cfaf8c'),this.scene,[-.2,-.008,.5]);ring.rotation.x=-Math.PI/2;ring.scale.y=.75;}

    // Amp
    const amp=new T.Group();amp.position.set(-2.6,.44,-.85);this.scene.add(amp);
    this.box([.82,.88,.55],'#23212b',amp,[0,0,0]);this.box([.69,.64,.025],'#5b5262',amp,[0,-.06,.29]);
    for(let i=0;i<9;i++)this.box([.01,.61,.014],'#322e3e',amp,[-.30+i*.075,-.06,.31]);
    for(let i=0;i<3;i++)this.ellipsoid([.025,.025,.02],'#dfb152',amp,[-.22+i*.16,.34,.31],0);
    const brand=this.label('FLY / AMP',.57,.14);brand.position.set(0,-.08,.32);amp.add(brand);
    const cable=new T.CatmullRomCurve3([new T.Vector3(-2.6,.03,-.55),new T.Vector3(-2.8,.03,.2),new T.Vector3(-1.7,.03,1.65),new T.Vector3(-.5,.07,1.3),new T.Vector3(-.85,.65,1.2)]);
    this.mesh(new T.TubeGeometry(cable,32,.019,5,false),this.material('#2c2635'));

    // Rincón dieciochero: Tonel de roble con Terremoto y Jarro de Greda
    const barrel=new T.Group();barrel.position.set(-3.1,.42,.8);this.scene.add(barrel);
    this.mesh(new T.CylinderGeometry(.34,.34,.84,14),this.material('#5a3922',{roughness:.8}),barrel,[0,0,0]);
    this.ellipsoid([.37,.43,.37],this.material('#664127',{roughness:.8}),barrel,[0,0,0],1);
    for(let y of [-.32,-.12,.12,.32]){
      const bRing=this.mesh(new T.TorusGeometry(.35,.015,4,14),this.material('#222126',{metalness:.6,roughness:.4}),barrel,[0,y,0]);
      bRing.rotation.x=Math.PI/2;
    }
    this.box([.74,.035,.74],'#6b452b',barrel,[0,.44,0]);

    // Vaso de Terremoto
    const terremoto=new T.Group();terremoto.position.set(-.12,.46,.08);barrel.add(terremoto);
    this.mesh(new T.CylinderGeometry(.095,.075,.28,12),this.material('#ffffff',{transparent:true,opacity:.42,roughness:.1}),terremoto,[0,.14,0]);
    this.mesh(new T.CylinderGeometry(.078,.073,.07,12),this.material('#c8102e',{roughness:.2}),terremoto,[0,.035,0]);
    this.mesh(new T.CylinderGeometry(.088,.078,.14,12),this.material('#eed26d',{roughness:.2,transparent:true,opacity:.85}),terremoto,[0,.14,0]);
    this.ellipsoid([.09,.08,.09],this.material('#fffde4',{roughness:.7}),terremoto,[0,.26,0],1);
    this.rod([.01,.18,0],[.07,.38,.04],.009,'#d52b1e',terremoto);
    this.rod([.02,.23,.01],[.07,.38,.04],.008,'#0033a0',terremoto);

    // Jarro de Greda chilena de Pomaire
    const greda=new T.Group();greda.position.set(.14,.46,-.06);barrel.add(greda);
    this.mesh(new T.CylinderGeometry(.08,.11,.22,12),this.material('#a85232',{roughness:.85}),greda,[0,.11,0]);
    this.ellipsoid([.13,.12,.13],this.material('#a85232',{roughness:.85}),greda,[0,.09,0],1);
    this.mesh(new T.TorusGeometry(.06,.016,4,8),this.material('#a85232',{roughness:.85}),greda,[.11,.11,0]);

    // Empanada de pino
    const empanada=this.ellipsoid([.12,.045,.075],this.material('#d99c43',{roughness:.6}),barrel,[0,.48,-.18],1);
    empanada.rotation.y=.4;

    // Floor lamp & plant
    this.rod([2.9,0,-2],[2.9,3.15,-2],.035,'#8b644d');
    this.mesh(new T.CylinderGeometry(.17,.55,.52,8),this.material('#e9c9ad'),this.scene,[2.9,3,-2]);
    this.mesh(new T.CylinderGeometry(.2,.16,.35,7),this.material('#a85232'),this.scene,[-3.15,.16,-2.25]);
    for(let i=0;i<5;i++){const a=i*2.4;const tip=[-3.15+Math.cos(a)*.3,.7+(i%2)*.3,-2.25+Math.sin(a)*.22];this.rod([-3.15,.3,-2.25],tip,.016,'#688a7f');const leaf=this.ellipsoid([.13,.28,.06],i%2?'#adc6a2':'#6e9a88',this.scene,tip,0);leaf.rotation.z=Math.cos(a)*.7;}
  }
  buildTV(){
    this.tv=new T.Group();this.tv.position.set(.95,1.85,-1.28);this.tv.rotation.y=-.12;this.scene.add(this.tv);
    this.box([3.6,2.53,.42],'#302d3c',this.tv,[0,0,0]);
    this.box([3.42,2.35,.06],'#857887',this.tv,[0,0,.235]);
    this.box([3.3,2.2,.05],'#141622',this.tv,[0,.035,.276]);
    this.textureCanvas=document.createElement('canvas');this.textureCanvas.width=960;this.textureCanvas.height=640;this.ctx=this.textureCanvas.getContext('2d');
    this.texture=new T.CanvasTexture(this.textureCanvas);this.texture.colorSpace=T.SRGBColorSpace;this.texture.minFilter=T.LinearFilter;
    const display=this.mesh(new T.PlaneGeometry(3.18,2.12),new T.MeshBasicMaterial({map:this.texture,toneMapped:false}),this.tv,[0,.035,.309]);display.castShadow=false;
    const logo=this.label('C U E C A   /   V I S I O N',1.2,.09);logo.position.set(-.1,-1.16,.275);this.tv.add(logo);
    this.ellipsoid([.025,.025,.02],this.material('#85ffcb',{emissive:'#85ffcb',emissiveIntensity:1}),this.tv,[1.48,-1.16,.28],0);
    for(const x of [-1.1,1.1]){this.rod([x,-1.25,0],[x-.18,-1.67,.35],.055,'#282432',this.tv);}
    this.box([3.95,.16,.95],'#aa8b80',this.scene,[.95,.18,-1.25]);
    for(const x of [-.6,2.5])this.box([.12,.15,.65],'#463b45',this.scene,[x,.03,-1.25]);
  }
  buildFly(){
    this.fly=new T.Group();this.fly.position.set(-1.28,0,1);this.fly.rotation.y=.12;this.scene.add(this.fly);
    this.body=new T.Group();this.body.position.y=1.05;this.fly.add(this.body);
    this.ellipsoid([.64,.33,.36],'#343b43',this.body,[-.43,0,0]);
    for(let i=0;i<4;i++){const band=this.mesh(new T.TorusGeometry(.29-i*.025,.022,4,9),this.material('#8c8074'),this.body,[-.32-i*.17,0,0]);band.rotation.y=Math.PI/2;band.scale.y=.94;}
    this.ellipsoid([.37,.37,.34],'#62685d',this.body,[.16,.07,0]);
    this.head=this.ellipsoid([.29,.31,.31],'#888172',this.body,[.53,.17,0]);
    for(const z of [-.235,.235]){
      this.ellipsoid([.225,.265,.15],this.material('#b9374e',{roughness:.38}),this.body,[.57,.19,z],2);
      this.ellipsoid([.06,.085,.018],'#f4bdac',this.body,[.62,.29,z+Math.sign(z)*.13],0);
      this.rod([.66,.4,z*.6],[.86,.68,z*.9],.014,'#3b3438',this.body);
      this.ellipsoid([.035,.045,.03],'#4a4243',this.body,[.86,.68,z*.9],0);
    }

    // Chupalla de huaso en la cabeza de la mosca
    const chupalla=new T.Group();chupalla.position.set(.54,.45,0);chupalla.rotation.set(.06,0,-.16);this.body.add(chupalla);
    this.mesh(new T.CylinderGeometry(.35,.35,.016,18),this.material('#dfc28d',{roughness:.8}),chupalla,[0,0,0]);
    this.mesh(new T.CylinderGeometry(.19,.20,.13,18),this.material('#cca66d',{roughness:.8}),chupalla,[0,.07,0]);
    this.mesh(new T.CylinderGeometry(.204,.204,.032,18),this.material('#18161b',{roughness:.5}),chupalla,[0,.024,0]);

    // Pañuelo de cueca chileno ondeando con el zapateo
    this.panuelo=new T.Group();this.panuelo.position.set(.56,.22,.18);this.body.add(this.panuelo);
    const pShape=new T.Shape();
    pShape.moveTo(0,0);pShape.lineTo(.16,.26);pShape.lineTo(.32,.12);pShape.lineTo(.22,-.16);pShape.lineTo(-.04,-.08);pShape.closePath();
    this.mesh(new T.ShapeGeometry(pShape),this.material('#ffffff',{roughness:.3,side:T.DoubleSide}),this.panuelo);
    this.rod([0,0,0],[.08,.12,.02],.012,'#e2e2e2',this.panuelo);

    // Two translucent wings
    this.wings=[];
    for(const side of [-1,1]){
      const wing=new T.Group();wing.position.set(.05,.29,side*.13);this.body.add(wing);this.wings.push(wing);
      const outline=[new T.Vector2(0,0),new T.Vector2(-.43,.28),new T.Vector2(-1.18,.57),new T.Vector2(-1.48,.49),new T.Vector2(-1.35,.22),new T.Vector2(-.58,-.06)];
      const shape=new T.Shape(outline);const m=this.mesh(new T.ShapeGeometry(shape),this.material('#cfddd9',{transparent:true,opacity:.68,side:T.DoubleSide,roughness:.3,depthWrite:false}),wing);m.rotation.x=side*Math.PI/2;
      for(const end of [[-1.39,.44],[-1.23,.25],[-.9,.43]])this.rod([0,.004,0],[end[0],.004,side*end[1]],.009,'#97aca9',wing);
      wing.rotation.x=side*.12;
    }
    for(let i=0;i<14;i++){const x=-.65+i*.09;this.rod([x,.29,0],[x-.03,.40+(i%3)*.02,0],.009,'#353838',this.body);}

    // Patas traseras plantadas de soporte (T3-L y T3-R)
    for(const side of [-1,1]){
      const root=[-.32,1.02,side*.20],knee=[-.62,.58,side*.55],ankle=[-.40,.08,side*.72];
      this.rod(root,knee,.036,'#555852',this.fly);this.rod(knee,ankle,.026,'#343d3c',this.fly);
      this.rod(ankle,[ankle[0]+.22,.05,ankle[2]+.03],.026,'#343d3c',this.fly);
      this.ellipsoid([.052,.052,.052],'#939183',this.fly,knee,0);
    }

    // 4 BALDOSAS DE ZAPATEO CUEQUERO EN EL SUELO (D, F, J, K)
    this.pads=[];
    const padColors=['#c292ff','#75d0b1','#f0bd70','#e88e9c'];
    const padLabels=['D','F','J','K'];
    const padPositions=[
      [.44,.02,-.36],
      [.08,.02,-.44],
      [.08,.02,.44],
      [.44,.02,.36]
    ];
    for(let i=0;i<4;i++){
      const pGroup=new T.Group();pGroup.position.set(...padPositions[i]);this.fly.add(pGroup);
      this.mesh(new T.CylinderGeometry(.16,.175,.035,14),this.material('#3d2e24',{roughness:.85}),pGroup,[0,.017,0]);
      this.mesh(new T.TorusGeometry(.155,.01,4,16),this.material('#b8860b',{metalness:.7,roughness:.3}),pGroup,[0,.035,0]);
      const padMat=this.material(padColors[i],{roughness:.25,metalness:.15,emissive:padColors[i],emissiveIntensity:.08});
      const padMesh=this.mesh(new T.CylinderGeometry(.135,.135,.01,14),padMat,pGroup,[0,.036,0]);
      this.pads.push(padMesh);
      const lbl=this.label(padLabels[i],.18,.18,'#ffffff');
      lbl.position.set(0,.042,0);lbl.rotation.x=-Math.PI/2;lbl.rotation.z=Math.PI/2;pGroup.add(lbl);
    }

    // 4 PATAS ARTICULADAS DE ZAPATEO (T1-L, T2-L, T2-R, T1-R)
    this.legs=[];
    const legConfigs=[
      {root:[.36,1.02,-.16],knee:[.52,.62,-.42],pad:[.44,.04,-.36]},
      {root:[.12,1.00,-.22],knee:[.02,.60,-.52],pad:[.08,.04,-.44]},
      {root:[.12,1.00,.22],knee:[.02,.60,.52],pad:[.08,.04,.44]},
      {root:[.36,1.02,.16],knee:[.52,.62,.42],pad:[.44,.04,.36]}
    ];
    for(let i=0;i<4;i++){
      const cfg=legConfigs[i];const legGroup=new T.Group();legGroup.position.set(cfg.root[0],cfg.root[1],cfg.root[2]);this.fly.add(legGroup);
      const kRel=[cfg.knee[0]-cfg.root[0],cfg.knee[1]-cfg.root[1],cfg.knee[2]-cfg.root[2]];
      const pRel=[cfg.pad[0]-cfg.root[0],cfg.pad[1]-cfg.root[1],cfg.pad[2]-cfg.root[2]];
      this.rod([0,0,0],kRel,.034,'#606860',legGroup);
      this.ellipsoid([.048,.048,.048],'#b5ad9e',legGroup,kRel,0);
      this.rod(kRel,pRel,.026,'#505850',legGroup);
      this.ellipsoid([.036,.036,.036],'#b5ad9e',legGroup,pRel,0);
      this.rod(pRel,[pRel[0]+.08,pRel[1],pRel[2]+(i<2?-.04:.04)],.018,'#343d3c',legGroup);
      this.legs.push({group:legGroup,baseRootY:cfg.root[1]});
    }
  }
  setView(value){this.view=value==='tv'?1:0;this.angle=0;}
  update(state,image){
    const reset=this.state&&(state.game.time<this.state.game.time||state.game.time===0);
    if(reset){this.event=null;this.eventUntil=0;this.lastCombo=0;this.comboBanner=null;}
    const event=state.game.events.at(-1),key=event?`${state.game.time}:${event.kind}:${event.lane}`:'';
    if(event&&key!==this.eventKey){
      this.eventKey=key;this.event=event;this.eventUntil=performance.now()+650;
      if(['miss','empty'].includes(event.kind) && (this.state?.game?.combo ?? 0) >= 6){
        this.comboBreakUntil = performance.now() + 700;
      }
    }
    const curCombo = state.game.combo ?? 0;
    if (this.lastCombo !== undefined) {
      if (curCombo >= 8 && this.lastCombo < 8) this.triggerComboBanner('¡MULTIPLICADOR 2×!', '#10b981');
      else if (curCombo >= 16 && this.lastCombo < 16) this.triggerComboBanner('¡MULTIPLICADOR 3×!', '#f59e0b');
      else if (curCombo >= 24 && this.lastCombo < 24) this.triggerComboBanner('¡4× EN LLAMAS! 🔥', '#ef4444');
      else if (curCombo >= 40 && this.lastCombo < 40) this.triggerComboBanner('¡RACHA PERFECTA! ★', '#a855f7');
    }
    this.lastCombo = curCombo;
    this.state=state;this.lastImage=image;this.paintScreen();
  }
  triggerComboBanner(text, color) {
    this.comboBanner = { text, color, until: performance.now() + 1100 };
  }
  drawFruitField(c, now) {
    // 1. Cielo soleado del campo chileno
    const sky = c.createLinearGradient(0, 0, 0, 220);
    sky.addColorStop(0, '#2563eb');
    sky.addColorStop(0.55, '#60a5fa');
    sky.addColorStop(0.9, '#fed7aa');
    sky.addColorStop(1, '#fef08a');
    c.fillStyle = sky;
    c.fillRect(0, 0, 960, 240);

    // Sol radiante en el horizonte
    c.fillStyle = '#fffbeb';
    c.beginPath(); c.arc(480, 85, 34, 0, Math.PI * 2); c.fill();
    c.fillStyle = 'rgba(254, 240, 138, 0.28)';
    c.beginPath(); c.arc(480, 85, 62, 0, Math.PI * 2); c.fill();

    // Silueta de la Cordillera de los Andes con picos nevados
    c.fillStyle = '#475569';
    c.beginPath();
    c.moveTo(0, 160);
    const mtn = [[80,95],[170,140],[260,80],[350,135],[440,75],[520,130],[610,70],[700,125],[790,85],[880,130],[960,105],[960,240],[0,240]];
    for (const pt of mtn) c.lineTo(pt[0], pt[1]);
    c.closePath(); c.fill();

    // Nieve en las cumbres
    c.fillStyle = '#f8fafc';
    for (const peak of [[80,95],[260,80],[440,75],[610,70],[790,85]]) {
      c.beginPath();
      c.moveTo(peak[0], peak[1]);
      c.lineTo(peak[0] - 22, peak[1] + 25);
      c.lineTo(peak[0], peak[1] + 18);
      c.lineTo(peak[0] + 22, peak[1] + 25);
      c.closePath(); c.fill();
    }

    // Suelo fértil de huerto frutal (tierra arada y pasto)
    const earth = c.createLinearGradient(0, 200, 0, 640);
    earth.addColorStop(0, '#365314');
    earth.addColorStop(0.3, '#2d4511');
    earth.addColorStop(1, '#1e2912');
    c.fillStyle = earth;
    c.fillRect(0, 200, 960, 440);

    // 2. Parrones de uva y árboles frutales a la izquierda
    this.drawOrchardSide(c, 0, 360, true, now);

    // 3. Parrones de uva y árboles frutales a la derecha
    this.drawOrchardSide(c, 600, 960, false, now);
  }
  drawOrchardSide(c, xMin, xMax, isLeft, now) {
    const dir = isLeft ? 1 : -1;
    const baseW = xMax - xMin;

    // Postes y vigas de parrón de madera rústica
    c.strokeStyle = '#78350f';
    c.lineWidth = 7;
    c.beginPath();
    const px1 = isLeft ? 60 : 900, px2 = isLeft ? 260 : 700;
    c.moveTo(px1, 230); c.lineTo(px1, 580);
    c.moveTo(px2, 210); c.lineTo(px2, 540);
    c.moveTo(px1 - 20, 240); c.lineTo(px2 + 20, 220);
    c.stroke();

    // Parras verdes con racimos de uva colgando
    c.fillStyle = '#166534';
    for (let i = 0; i < 7; i++) {
      const vx = isLeft ? 30 + i * 45 : 680 + i * 42;
      const vy = 205 + (i % 3) * 35;
      c.beginPath(); c.arc(vx, vy, 28, 0, Math.PI * 2); c.fill();
      c.fillStyle = i % 2 ? '#22c55e' : '#15803d';
      c.beginPath(); c.arc(vx + 6, vy - 6, 22, 0, Math.PI * 2); c.fill();

      // Racimos de uvas moradas y uvas moscatel
      const isPurple = i % 2 === 0;
      c.fillStyle = isPurple ? '#7e22ce' : '#84cc16';
      for (let r = 0; r < 6; r++) {
        const gx = vx + ((r % 3) - 1) * 7;
        const gy = vy + 24 + Math.floor(r / 3) * 9;
        c.beginPath(); c.arc(gx, gy, 5.5, 0, Math.PI * 2); c.fill();
      }
    }

    // Árboles de duraznos / melocotones en primer plano
    const tx = isLeft ? 110 : 850, ty = 390;
    c.fillStyle = '#451a03';
    c.fillRect(tx - 12, ty, 24, 160);
    c.fillStyle = '#15803d';
    c.beginPath(); c.arc(tx, ty, 68, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#16a34a';
    c.beginPath(); c.arc(tx - 14, ty - 12, 54, 0, Math.PI * 2); c.fill();

    // Duraznos maduros con rubor rojo/naranja
    const peaches = isLeft ? [[-35,-20],[25,-30],[-15,22],[34,14],[-2,40],[18,-8]] : [[-25,-25],[30,-15],[-20,20],[25,18],[0,35],[-15,-5]];
    for (const p of peaches) {
      c.fillStyle = '#f97316';
      c.beginPath(); c.arc(tx + p[0], ty + p[1], 9, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#dc2626';
      c.beginPath(); c.arc(tx + p[0] + 2, ty + p[1] - 2, 5, 0, Math.PI * 2); c.fill();
    }

    // Cajón cosechero de madera con frutas en la esquina inferior
    const bx = isLeft ? 30 : 830, by = 550;
    c.fillStyle = '#92400e';
    c.fillRect(bx, by, 95, 52);
    c.strokeStyle = '#b45309'; c.lineWidth = 3;
    c.strokeRect(bx, by, 95, 52);
    for (let l = 1; l <= 2; l++) {
      c.beginPath(); c.moveTo(bx, by + l * 17); c.lineTo(bx + 95, by + l * 17); c.stroke();
    }
    // Frutas en el cajón
    for (let f = 0; f < 8; f++) {
      c.fillStyle = f % 2 ? '#ea580c' : '#a855f7';
      c.beginPath(); c.arc(bx + 12 + f * 10.5, by - 2 + (f % 2) * 4, 7, 0, Math.PI * 2); c.fill();
    }
  }
  paintScreen(){
    const c=this.ctx,s=this.state,now=performance.now();
    const song = s?.game?.song;
    const songTitle = song?.title || 'LA CONSENTIDA';
    const songSub = (song?.subtitle || 'CUECA TRADICIONAL CHILENA').toUpperCase() + ' · 6/8';

    // 1. Campo de frutas para la mosca de la fruta
    this.drawFruitField(c, now);

    // 2. Reproyección 3D de la pista sobre el camino del huerto
    if(this.lastImage){
      for(let y=0;y<400;y+=2){
        const t=y/400,width=235+505*t;
        c.drawImage(this.lastImage,0,y,320,2,480-width/2,95+t*490,width,3.5);
      }
    }

    // Bordes rústicos de madera noble para la autopista
    c.strokeStyle='#dfb152';c.lineWidth=4;c.beginPath();
    c.moveTo(362,95);c.lineTo(110,585);c.moveTo(598,95);c.lineTo(850,585);c.stroke();
    c.strokeStyle='rgba(255,255,255,0.4)';c.lineWidth=1.5;c.beginPath();
    c.moveTo(363,95);c.lineTo(112,585);c.moveTo(597,95);c.lineTo(848,585);c.stroke();

    // 3. Letras receptoras iluminadas D, F, J, K
    const colors=['#c292ff','#75d0b1','#f0bd70','#e88e9c'];
    c.font='bold 24px monospace';c.textAlign='center';
    for(let i=0;i<4;i++){
      const active = Boolean(s?.game?.action?.[i] || s?.game?.active_holds?.[i]);
      c.fillStyle = active ? '#ffffff' : colors[i];
      if (active) {
        c.shadowColor = colors[i]; c.shadowBlur = 12;
      }
      c.fillText('DFJK'[i], 202+i*185, 616);
      c.shadowBlur = 0;
    }

    // 4. Cabecera dinámica de la canción
    c.textAlign='left';
    c.fillStyle='rgba(15, 23, 42, 0.72)';
    c.fillRect(16, 12, 420, 68);
    c.strokeStyle='#e2b765'; c.lineWidth=2;
    c.strokeRect(16, 12, 420, 68);
    c.fillStyle='#fef08a';c.font='bold 22px monospace';c.fillText(`♪ ${songTitle}`, 32, 42);
    c.font='15px monospace';c.fillStyle='#e2e8f0';c.fillText(songSub, 32, 66);

    // Puntuación en esquina superior derecha
    c.fillStyle='rgba(15, 23, 42, 0.72)';
    c.fillRect(660, 12, 284, 68);
    c.strokeStyle='#e2b765'; c.lineWidth=2;
    c.strokeRect(660, 12, 284, 68);
    c.textAlign='right';c.fillStyle='#fff0d8';c.font='bold 34px monospace';
    c.fillText(String(s?.game.score??0).padStart(5,'0'), 925, 57);
    c.textAlign='left';

    // 5. HUD Épico de Combo estilo Guitar Hero
    const combo = s?.game?.combo ?? 0;
    const mult = combo >= 24 ? 4 : combo >= 16 ? 3 : combo >= 8 ? 2 : 1;
    const multColors = { 1: '#94a3b8', 2: '#10b981', 3: '#f59e0b', 4: '#ef4444' };
    const multCol = multColors[mult];

    // Dial circular de Multiplicador en el lateral izquierdo
    const dialX = 64, dialY = 220;
    c.save();
    c.fillStyle = 'rgba(15, 23, 42, 0.85)';
    c.beginPath(); c.arc(dialX, dialY, 44, 0, Math.PI * 2); c.fill();
    c.lineWidth = mult === 4 ? 6 : 4;
    c.strokeStyle = multCol;
    if (mult === 4) {
      c.shadowColor = '#ef4444'; c.shadowBlur = 18;
    }
    c.stroke();

    // Corona de llamas animada para Multiplicador 4x
    if (mult === 4) {
      c.fillStyle = 'rgba(239, 68, 68, 0.35)';
      for (let f = 0; f < 6; f++) {
        const a = (f * Math.PI / 3) + Math.sin(now * 0.008 + f) * 0.2;
        const fx = dialX + Math.cos(a) * (48 + Math.sin(now * 0.015 + f) * 6);
        const fy = dialY + Math.sin(a) * (48 + Math.sin(now * 0.015 + f) * 6);
        c.beginPath(); c.arc(fx, fy, 8, 0, Math.PI * 2); c.fill();
      }
    }

    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = '#ffffff';
    c.font = 'bold 36px monospace';
    c.fillText(`${mult}×`, dialX, dialY - 2);

    c.font = 'bold 12px monospace';
    c.fillStyle = multCol;
    c.fillText(mult === 4 ? 'EN FUEGO' : 'MULT', dialX, dialY + 28);
    c.restore();

    // Barra de Combo al lado del dial
    c.fillStyle = 'rgba(15, 23, 42, 0.8)';
    c.fillRect(16, 280, 96, 54);
    c.strokeStyle = '#475569'; c.lineWidth = 1.5;
    c.strokeRect(16, 280, 96, 54);
    c.textAlign = 'center';
    c.fillStyle = '#e2e8f0';
    c.font = '13px monospace';
    c.fillText('COMBO', 64, 298);
    c.font = 'bold 22px monospace';
    c.fillStyle = mult === 4 ? '#ef4444' : '#fef08a';
    c.fillText(`${combo}`, 64, 322);

    // Contadores de aciertos y fallos
    c.textAlign = 'left';
    c.fillStyle = 'rgba(15, 23, 42, 0.8)';
    c.fillRect(16, 344, 96, 58);
    c.strokeStyle = '#475569'; c.lineWidth = 1.5;
    c.strokeRect(16, 344, 96, 58);
    c.font = '14px monospace';
    c.fillStyle = '#6ee7b7'; c.fillText(`✓ ${s?.game.hits??0}`, 26, 368);
    c.fillStyle = '#fca5a5'; c.fillText(`× ${s?.game.misses??0}`, 26, 390);

    // 6. Juicios de impacto (PERFECTO / BIEN / SE ESCAPÓ)
    if(this.event && now < this.eventUntil){
      c.save();
      c.textAlign = 'center';
      const isBad = ['miss','empty'].includes(this.event.kind);
      c.fillStyle = isBad ? '#ef4444' : '#10b981';
      c.shadowColor = isBad ? '#ef4444' : '#34d399';
      c.shadowBlur = 14;
      c.font = 'bold 36px monospace';
      c.fillText({perfect:'★ ¡PERFECTO! ★', good:'¡BIEN!', miss:'SE ESCAPÓ', empty:'FUERA DE TIEMPO'}[this.event.kind], 480, 190);
      c.restore();
    }

    // 7. Cartel explosivo de hito de combo
    if (this.comboBanner && now < this.comboBanner.until) {
      c.save();
      c.textAlign = 'center';
      c.fillStyle = 'rgba(15, 23, 42, 0.88)';
      c.fillRect(240, 220, 480, 62);
      c.strokeStyle = this.comboBanner.color;
      c.lineWidth = 3;
      c.shadowColor = this.comboBanner.color;
      c.shadowBlur = 20;
      c.strokeRect(240, 220, 480, 62);
      c.fillStyle = '#ffffff';
      c.font = 'bold 30px monospace';
      c.fillText(this.comboBanner.text, 480, 252);
      c.restore();
    }

    // Aviso de combo roto
    if (this.comboBreakUntil && now < this.comboBreakUntil) {
      c.save();
      c.textAlign = 'center';
      c.fillStyle = '#f87171';
      c.font = 'bold 22px monospace';
      c.fillText('¡COMBO ROTO!', 480, 290);
      c.restore();
    }

    if(s && s.mode !== 'running'){
      c.textAlign = 'center';
      c.fillStyle = 'rgba(15, 23, 42, 0.85)';
      c.fillRect(320, 96, 320, 36);
      c.strokeStyle = '#e2b765'; c.lineWidth = 1.5; c.strokeRect(320, 96, 320, 36);
      c.fillStyle = '#fef08a'; c.font = 'bold 16px monospace';
      c.fillText(s.mode==='finished'?'SESIÓN COMPLETA':s.mode==='loading'?'CARGANDO MALECNS':s.game.time>0?'EN PAUSA':'LISTO PARA TOCAR', 480, 114);
    }
    c.textAlign = 'left';
    this.texture.needsUpdate = true;
  }
  render(now){
    if(now-this.lastRender<1000/30)return;const dt=Number.isFinite(this.lastRender)?Math.min((now-this.lastRender)/1000,.1):0;this.lastRender=now;
    this.animationTime=(this.animationTime||0)+(this.state?.mode==='running'?dt:0);
    const canvas=this.renderer.domElement,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
    if(w!==this.w||h!==this.h){this.w=w;this.h=h;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
    const s=this.state,t=this.animationTime||0,playing=s?.mode==='running',action=s?.game.action.some(Boolean),motion=!this.reduced.matches;
    this.body.position.y=1.05+(motion&&playing?Math.sin(t*8)*.024:0);
    this.panuelo.rotation.z=motion&&playing?Math.sin(t*12)*.45:0;
    this.panuelo.rotation.y=motion&&playing?Math.cos(t*10)*.35:0;
    
    // Animación de las patas articuladas y soporte de sustain / hold
    for(let i=0;i<4;i++){
      const active=Boolean(s?.game?.action?.[i]);
      const holding=Boolean(s?.game?.active_holds?.[i]);
      const leg=this.legs[i];
      // Si está en hold sostenido, la pata permanece firmemente pisada en la baldosa
      const stomp = holding ? 0.0 : (motion && active ? Math.max(0, Math.sin(t * 45)) * 0.14 : 0);
      leg.group.position.y = leg.baseRootY + stomp;
      leg.group.rotation.x = holding ? 0.04 : (motion && active ? Math.sin(t * 45) * 0.12 : 0);
      this.pads[i].material.emissiveIntensity = holding ? 2.2 : (active ? 1.0 : 0.08);
    }
    this.wings.forEach((wing,i)=>wing.rotation.x=(i?1:-1)*(.12+(motion&&playing?Math.sin(t*12)*.035:0)));
    const a=this.angle;
    if(this.view){this.camera.position.set(.9+Math.sin(a)*3,2.6,5.6);this.camera.lookAt(.9,1.8,-1.2);}
    else{this.camera.position.set(4.2+Math.sin(a)*6,4.4,8.5+Math.cos(a)*.5);this.camera.lookAt(-.05,1.35,-.1);}
    const failed=this.event&&['miss','empty'].includes(this.event.kind)&&now<this.eventUntil;
    const success=this.event&&['good','perfect'].includes(this.event.kind)&&now<this.eventUntil;
    const curCombo = s?.game?.combo ?? 0;
    const isFire = curCombo >= 24;
    this.screenLight.color.set(failed ? '#dc788b' : isFire ? '#ff6622' : success ? '#8fdfbb' : '#b7a0ff');
    if(this.eventUntil&&now>this.eventUntil){this.eventUntil=0;this.paintScreen();}
    this.renderer.render(this.scene,this.camera);
  }
}
